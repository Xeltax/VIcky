//! Translated shaders kept on disk (decision 0003): under the configured
//! cache folder, keyed by the container's content and by the translator
//! that translated it (its source fingerprint and settings, see `build`), so
//! a changed translator never serves an old translation. Entries go through
//! `victory-pipeline`'s content-addressed cache, which writes them
//! atomically; an entry that does not read back is translated again.
//!
//! The entry is our own encoding of [`HostShader`]: a field added to it
//! must be added to `encode` and `decode` too (entries of older builds
//! are ignored anyway).

use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use victory_pipeline::{Cache, Converter};
use victory_shader::emit::{HostShader, StaticSampler, UserData};
use victory_shader::resource::RootIndexedView;
use victory_shader::{HwStage, RootType};

static CACHE: OnceLock<Cache> = OnceLock::new();
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

/// Keeps translated shaders in `dir` from now on.
pub fn use_shader_cache(dir: PathBuf) {
    match Cache::open(dir) {
        Ok(cache) => {
            let _ = CACHE.set(cache);
        }
        Err(e) => tracing::warn!(error = %e, "no shader cache"),
    }
}

/// Shaders found in the cache and translated since the start.
pub(crate) fn counts() -> (u64, u64) {
    (HITS.load(Ordering::Relaxed), MISSES.load(Ordering::Relaxed))
}

/// What identifies the translation, besides the container: the shader
/// crate's fingerprint (its source, every switch it reads, the host facts the
/// device gave it, the loop limit, barrier waits and clean-up passes) and
/// this file's encoding of the result. Not the whole executable any more: a
/// rebuild that leaves the translator alone keeps its translations (each new
/// build used to translate the title's thousands of shaders again, which kept
/// the first minutes of every test run short of pipelines).
/// `VICTORY_SHADER_CACHE_KEY=exe` restores the old key (the executable's hash).
/// Worked out per lookup, so a translation made before the device said its
/// wave size is never filed under the later key.
fn build() -> [u8; 32] {
    static WARNED: std::sync::Once = std::sync::Once::new();
    let limit = victory_shader::emit::loop_limit();
    if limit != victory_shader::emit::DEFAULT_LOOP_LIMIT {
        WARNED.call_once(|| tracing::warn!(limit, "VICTORY_SHADER_LOOP_LIMIT: translated loops stop sooner"));
    }
    static EXE: OnceLock<Option<[u8; 32]>> = OnceLock::new();
    let exe = EXE.get_or_init(|| {
        if std::env::var_os("VICTORY_SHADER_CACHE_KEY").is_some_and(|v| v == "exe") {
            match std::env::current_exe().and_then(std::fs::read) {
                Ok(bytes) => Some(*blake3::hash(&bytes).as_bytes()),
                Err(e) => {
                    tracing::warn!(error = %e, "cannot read our executable; keying on the translator");
                    None
                }
            }
        } else {
            None
        }
    });
    let mut h = blake3::Hasher::new();
    h.update(b"victory shader cache key 2\0");
    if let Some(exe) = exe {
        h.update(exe);
    }
    h.update(&victory_shader::emit::translation_fingerprint());
    h.update(include_bytes!("shader_cache.rs"));
    *h.finalize().as_bytes()
}

/// The offline warm-up (`victory translate --warm`): `container` is
/// translated into the cache unless it is there already. `Ok(true)`: it
/// was translated now. Needs [`use_shader_cache`] and the same host facts
/// the run will have (`victory_shader::emit::translation_fingerprint`).
pub fn warm(container: &[u8]) -> Result<bool, String> {
    let cache = CACHE.get().ok_or("no shader cache")?;
    // The heap map of a capped host, as its run translates with it.
    crate::descriptor::host::offline_heap_map();
    let ran = AtomicBool::new(false);
    // As at run time ([`host_shader`]): by its hash and under its own keys.
    translation(Some(cache), container, &ran).map(|_| ran.load(Ordering::Relaxed))
}

/// The translation of a console shader container, from the cache when
/// there is one.
pub(crate) fn host_shader(container: &[u8]) -> Result<HostShader, String> {
    translation(CACHE.get(), container, &AtomicBool::new(false))
}

/// The translation of `container` through `cache` if any, by the hash the
/// log names the shader by (`pipeline.rs`), which the per-program switches
/// take; `ran` is set when some attempt translated (was not in the cache).
///
/// With `VICTORY_WAVE_MASK_COPIES` at `retry` (the default) a console
/// wave64 program refused without the mask copies is translated again with
/// them (`victory_shader::emit::MaskCopies`). Each attempt runs with the
/// copies chosen here (`emit::with_mask_copies`, so the translator makes no
/// second attempt of its own) under the key of that choice
/// ([`program_key`]): the second attempt is filed where `all` files that
/// program, never where the first attempt would be. A refusal is not
/// cached, so such a program's first attempt runs once a run.
fn translation(cache: Option<&Cache>, container: &[u8], ran: &AtomicBool) -> Result<HostShader, String> {
    use victory_shader::emit;
    let hash = blake3::hash(container).to_hex()[..16].to_owned();
    let attempt = |copies: bool| {
        let translate = |c: &[u8]| {
            ran.store(true, Ordering::Relaxed);
            emit::for_program(&hash, || emit::with_mask_copies(copies, || emit::translate(c))).map_err(|e| e.to_string())
        };
        match cache {
            Some(cache) => {
                // (`wave64_retried` second: it parses the container, and only a
                // program a list names, or with `VICTORY_WAVE_DEEP_SKIPS=0`
                // any, gets there)
                let off = !emit::deep_skips_allowed(Some(&hash)) && emit::wave64_retried(container);
                let dropped = emit::dropped_stores(Some(&hash));
                // (`wave64_program` second, as above)
                let pairs = emit::site_pairs_allowed(Some(&hash)) && emit::wave64_program(container);
                // (`wave64_program` second, as above)
                let paired = (emit::wave_pairs_allowed(Some(&hash)) && emit::wave64_program(container))
                    .then(emit::wave_pairs_settings);
                let flags = emit::lds_flags_allowed(Some(&hash)) && emit::wave64_program(container);
                let key = program_key(&build(), off, opt_off(&hash), copies, pairs, flags, paired.as_deref(), &dropped);
                cached(cache, &key, container, translate)
            }
            None => translate(container),
        }
    };
    let copied = attempts(emit::mask_copies(Some(&hash)), || emit::wave64_program(container), attempt);
    if copied.1 {
        tracing::info!(hash = %hash, ok = copied.0.is_ok(), "shader refused without the mask copies, translated again with them");
    }
    copied.0
}

/// The attempts [`translation`] makes under `mode`: `attempt(copies)` once,
/// or for `MaskCopies::Retry` once more with the copies after a refusal
/// that asks for it (`wave64`: whether the program is a console wave64 one,
/// asked only then). The result, and whether there was a second attempt.
fn attempts<T>(
    mode: victory_shader::emit::MaskCopies,
    wave64: impl Fn() -> bool,
    attempt: impl Fn(bool) -> Result<T, String>,
) -> (Result<T, String>, bool) {
    use victory_shader::emit::{self, MaskCopies};
    match mode {
        MaskCopies::Off => (attempt(false), false),
        MaskCopies::On => (attempt(wave64()), false),
        MaskCopies::Retry => match attempt(false) {
            Err(e) if emit::mask_copies_retried(&e) && wave64() => match attempt(true) {
                Ok(again) => (Ok(again), true),
                // (refused again: the first refusal, as without the switch)
                Err(_) => (Err(e), true),
            },
            first => (first, false),
        },
    }
}

/// The key of one program's entries: the build's, mixed with the
/// per-program decisions that differ from the build's own.
/// - The program is kept out of the wave64 form "without skips, scalars
///   followed" (`VICTORY_WAVE_DEEP_SKIPS_NOT`, `_ONLY`:
///   `victory_shader::emit::deep_skips_allowed`). Only for a wave64 compute
///   program on a host of 32-lane waves, the only ones it can change.
/// - The program is kept out of the clean-up passes while the build runs
///   some (`VICTORY_SHADER_OPT_NOT`, `_ONLY`:
///   `victory_shader::emit::shader_opt_allowed`).
/// - The translation takes a wave64 mask restored from scalar copies as the
///   mask it was copied from (`VICTORY_WAVE_MASK_COPIES`:
///   `victory_shader::emit::mask_copies`; by default only a second attempt
///   after a refusal, [`translation`]). Only for a console wave64 program,
///   the only ones it can change.
/// - The program is built with fork #1002's forms (`VICTORY_WAVE_SITE_PAIRS`:
///   `victory_shader::emit::site_pairs_allowed`). Only for a console wave64
///   program, the only ones they can change.
/// - The program takes LDS flag regions as two host waves (`VICTORY_LDS_FLAGS`:
///   `victory_shader::emit::lds_flags_allowed`). Only for a console wave64 program.
/// - The program tries the paired form first (`VICTORY_WAVE_PAIRS_ONLY`:
///   `victory_shader::emit::wave_pairs_allowed`), with the settings of that form
///   (`victory_shader::emit::wave_pairs_settings`). Only for a console wave64 program.
/// - Stores left out of the program (`VICTORY_STORE_DROP`, diagnostics:
///   `victory_shader::emit::dropped_stores`): their console offsets.
///
/// So the lists are in no key: changing one translates again only the
/// programs whose decision it changes, and a program the lists leave alone
/// keeps the key it had without any list.
#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)] // one per switch, each mixed in on its own
fn program_key(
    build: &[u8; 32],
    deep_skips_off: bool,
    opt_off: bool,
    mask_copies: bool,
    site_pairs: bool,
    lds_flags: bool,
    wave_pairs: Option<&str>,
    dropped: &[usize],
) -> [u8; 32] {
    if !deep_skips_off && !opt_off && !mask_copies && !site_pairs && !lds_flags && wave_pairs.is_none() && dropped.is_empty()
    {
        return *build;
    }
    let mut h = blake3::Hasher::new();
    h.update(build);
    if deep_skips_off {
        h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
    }
    if opt_off {
        h.update(b"VICTORY_SHADER_OPT off for this program");
    }
    if mask_copies {
        h.update(b"VICTORY_WAVE_MASK_COPIES on for this program");
    }
    if site_pairs {
        h.update(b"VICTORY_WAVE_SITE_PAIRS on for this program");
    }
    if lds_flags {
        h.update(b"VICTORY_LDS_FLAGS on for this program");
    }
    if let Some(settings) = wave_pairs {
        h.update(b"VICTORY_WAVE_PAIRS_ONLY on for this program:");
        h.update(settings.as_bytes());
    }
    if !dropped.is_empty() {
        h.update(b"VICTORY_STORE_DROP for this program:");
        for offset in dropped {
            h.update(&(*offset as u64).to_le_bytes());
        }
    }
    *h.finalize().as_bytes()
}

/// Whether the program `hash` is kept out of clean-up passes the build
/// runs (with none, the lists change nothing and are in no key).
fn opt_off(hash: &str) -> bool {
    use victory_shader::emit;
    emit::shader_opt() != 0 && !emit::shader_opt_allowed(Some(hash))
}

/// The cache entry's converter: the build and the container in, the
/// encoded translation out. `ran` tells a miss from a hit.
struct Translate<F> {
    translate: F,
    ran: AtomicBool,
}

impl<F: Fn(&[u8]) -> Result<HostShader, String> + Send + Sync> Converter for Translate<F> {
    fn id(&self) -> &'static str {
        "shader.dxil"
    }
    fn version(&self) -> u32 {
        // The build fingerprint in the input does the versioning.
        1
    }
    fn convert(&self, input: &[u8]) -> anyhow::Result<Vec<u8>> {
        self.ran.store(true, Ordering::Relaxed);
        let shader = (self.translate)(&input[32..]).map_err(anyhow::Error::msg)?;
        Ok(encode(&shader))
    }
}

fn cached(
    cache: &Cache,
    build: &[u8; 32],
    container: &[u8],
    translate: impl Fn(&[u8]) -> Result<HostShader, String> + Send + Sync,
) -> Result<HostShader, String> {
    let input = [&build[..], container].concat();
    let converter = Translate {
        translate,
        ran: AtomicBool::new(false),
    };
    for _ in 0..2 {
        let path = cache
            .get_or_convert(&converter, &input)
            .map_err(|e| format!("{e:#}"))?;
        if let Some(shader) = std::fs::read(&path).ok().and_then(|b| decode(&b)) {
            let (counter, what) = if converter.ran.load(Ordering::Relaxed) {
                (&MISSES, "translated and cached")
            } else {
                (&HITS, "from the cache")
            };
            counter.fetch_add(1, Ordering::Relaxed);
            tracing::debug!(path = %path.display(), "shader {what}");
            return Ok(shader);
        }
        tracing::debug!(path = %path.display(), "unreadable shader cache entry; translating again");
        let _ = std::fs::remove_file(&path);
    }
    MISSES.fetch_add(1, Ordering::Relaxed);
    (converter.translate)(container)
}

const MAGIC: &[u8; 4] = b"VSH1";
const STAGES: [HwStage; 8] = [
    HwStage::Ls,
    HwStage::Hs,
    HwStage::Es,
    HwStage::Gs,
    HwStage::Vs,
    HwStage::Ps,
    HwStage::Ms,
    HwStage::Cs,
];

fn root_code(r: RootType) -> u32 {
    match r {
        RootType::Constants => 0,
        RootType::Descriptor => 1,
        RootType::Table => 2,
        RootType::ImmediateSampler => 3,
        RootType::ResourceHeap => 4,
        RootType::SamplerHeap => 5,
        RootType::Other(x) => 6 + x,
    }
}

fn root_type(code: u32) -> RootType {
    match code {
        0 => RootType::Constants,
        1 => RootType::Descriptor,
        2 => RootType::Table,
        3 => RootType::ImmediateSampler,
        4 => RootType::ResourceHeap,
        5 => RootType::SamplerHeap,
        x => RootType::Other(x - 6),
    }
}

/// Little-endian words and length-prefixed lists.
fn encode(s: &HostShader) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    let word = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
    let stage = STAGES.iter().position(|&x| x == s.stage).unwrap_or(0) as u32;
    word(&mut out, stage);
    word(&mut out, s.dxil.len() as u32);
    out.extend(&s.dxil);
    word(&mut out, s.user_data.len() as u32);
    for u in &s.user_data {
        for v in [u.index, u32::from(u.slot), root_code(u.root), u.dword] {
            word(&mut out, v);
        }
    }
    word(&mut out, s.samplers.len() as u32);
    for sampler in &s.samplers {
        for &v in sampler {
            word(&mut out, v);
        }
    }
    word(&mut out, s.static_samplers.len() as u32);
    for t in &s.static_samplers {
        let [u, v, w] = t.address;
        let words = [
            t.filter,
            u,
            v,
            w,
            t.mip_lod_bias.to_bits(),
            t.max_anisotropy,
            t.comparison,
            t.border,
            t.min_lod.to_bits(),
            t.max_lod.to_bits(),
        ];
        for v in words {
            word(&mut out, v);
        }
    }
    for slots in [&s.root_cbvs, &s.root_srvs, &s.root_uavs] {
        word(&mut out, slots.len() as u32);
        out.extend(slots);
    }
    word(&mut out, s.root_indexed_views.len() as u32);
    for view in &s.root_indexed_views {
        for value in [
            u32::from(view.table_slot),
            u32::from(view.constant_slot),
            view.constant_dword,
            view.byte_offset,
            u32::from(view.shift),
        ] {
            word(&mut out, value);
        }
    }
    out
}

fn decode(bytes: &[u8]) -> Option<HostShader> {
    struct Reader<'a>(&'a [u8]);
    impl<'a> Reader<'a> {
        fn take(&mut self, n: usize) -> Option<&'a [u8]> {
            let (head, rest) = self.0.split_at_checked(n)?;
            self.0 = rest;
            Some(head)
        }
        fn word(&mut self) -> Option<u32> {
            Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
        }
        fn list(&mut self) -> Option<Vec<u8>> {
            let n = self.word()? as usize;
            Some(self.take(n)?.to_vec())
        }
    }
    let mut r = Reader(bytes);
    if r.take(4)? != MAGIC {
        return None;
    }
    let stage = *STAGES.get(r.word()? as usize)?;
    let dxil = r.list()?;
    let n = r.word()? as usize;
    let mut user_data = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        user_data.push(UserData {
            index: r.word()?,
            slot: u8::try_from(r.word()?).ok()?,
            root: root_type(r.word()?),
            dword: r.word()?,
        });
    }
    let n = r.word()? as usize;
    let mut samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        samplers.push([r.word()?, r.word()?, r.word()?, r.word()?]);
    }
    let n = r.word()? as usize;
    let mut static_samplers = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        static_samplers.push(StaticSampler {
            filter: r.word()?,
            address: [r.word()?, r.word()?, r.word()?],
            mip_lod_bias: f32::from_bits(r.word()?),
            max_anisotropy: r.word()?,
            comparison: r.word()?,
            border: r.word()?,
            min_lod: f32::from_bits(r.word()?),
            max_lod: f32::from_bits(r.word()?),
        });
    }
    let root_cbvs = r.list()?;
    let root_srvs = r.list()?;
    let root_uavs = r.list()?;
    let n = r.word()? as usize;
    let mut root_indexed_views = Vec::with_capacity(n.min(256));
    for _ in 0..n {
        root_indexed_views.push(RootIndexedView {
            table_slot: u8::try_from(r.word()?).ok()?,
            constant_slot: u8::try_from(r.word()?).ok()?,
            constant_dword: r.word()?,
            byte_offset: r.word()?,
            shift: u8::try_from(r.word()?).ok().filter(|&s| s < 32)?,
        });
    }
    r.0.is_empty().then_some(HostShader {
        stage,
        dxil,
        user_data,
        samplers,
        static_samplers,
        root_cbvs,
        root_srvs,
        root_uavs,
        root_indexed_views,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shader() -> HostShader {
        HostShader {
            stage: HwStage::Ps,
            dxil: b"DXBC and the rest".to_vec(),
            user_data: vec![UserData {
                index: 8,
                slot: 1,
                root: RootType::Table,
                dword: 0,
            }],
            samplers: vec![[0x400, 0xc0_0000, 0x18f0_0000, 0]],
            static_samplers: vec![StaticSampler::from_words([
                0x400,
                0xc0_0000,
                0x18f0_0000,
                0,
            ])],
            root_cbvs: vec![2],
            root_srvs: vec![0, 3],
            root_uavs: vec![1],
            root_indexed_views: vec![RootIndexedView {
                table_slot: 1,
                constant_slot: 2,
                constant_dword: 3,
                byte_offset: 32,
                shift: 5,
            }],
        }
    }

    #[test]
    fn translations_come_back_from_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let build = [7u8; 32];
        let container = b"a console container".as_slice();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 1, "the second came from disk");

        // Another build translates again.
        assert_eq!(cached(&cache, &[8; 32], container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 2);

        // A damaged entry is a miss, translated and stored again.
        for e in walk(dir.path()) {
            std::fs::write(e, b"VSH1 broken").unwrap();
        }
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        assert_eq!(cached(&cache, &build, container, translate), Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3);

        // Failures are not stored.
        let fail = |_: &[u8]| Err::<HostShader, _>("unsupported".to_owned());
        assert!(cached(&cache, &build, b"other", fail).is_err());
    }

    fn walk(dir: &std::path::Path) -> Vec<PathBuf> {
        let mut files = Vec::new();
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            if e.path().is_dir() {
                files.extend(walk(&e.path()));
            } else {
                files.push(e.path());
            }
        }
        files
    }

    #[test]
    fn a_program_kept_out_of_the_deep_skips_form_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(program_key(&build, false, false, false, false, false, None, &[]), build, "allowed: the build's key, as without lists");
        let off = program_key(&build, true, false, false, false, false, None, &[]);
        assert_ne!(off, build);
        assert_eq!(program_key(&build, true, false, false, false, false, None, &[]), off, "the same decision, the same key");
        assert_ne!(program_key(&[8; 32], true, false, false, false, false, None, &[]), off, "and still the build's");

        // So only the program whose decision changed is translated again.
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        let runs = AtomicU64::new(0);
        let translate = |_: &[u8]| {
            runs.fetch_add(1, Ordering::Relaxed);
            Ok(shader())
        };
        let [a, b] = [b"program a".as_slice(), b"program b"];
        for c in [a, b] {
            cached(&cache, &program_key(&build, false, false, false, false, false, None, &[]), c, translate).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // a listed: a translated again, b from the cache
        cached(&cache, &program_key(&build, true, false, false, false, false, None, &[]), a, translate).unwrap();
        cached(&cache, &program_key(&build, false, false, false, false, false, None, &[]), b, translate).unwrap();
        assert_eq!(runs.load(Ordering::Relaxed), 3);
        // the list emptied again: both from the cache
        for c in [a, b] {
            cached(&cache, &program_key(&build, false, false, false, false, false, None, &[]), c, translate).unwrap();
        }
        assert_eq!(runs.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_program_kept_out_of_the_clean_up_has_a_key_of_its_own() {
        let build = [7u8; 32];
        assert_eq!(program_key(&build, false, false, false, false, false, None, &[]), build);
        let opt = program_key(&build, false, true, false, false, false, None, &[]);
        let deep = program_key(&build, true, false, false, false, false, None, &[]);
        let both = program_key(&build, true, true, false, false, false, None, &[]);
        assert_ne!(opt, build);
        // Each decision a key of its own, the deep-skips one as before.
        assert!(opt != deep && opt != both && deep != both);
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
        let unchanged = *h.finalize().as_bytes();
        assert_eq!(deep, unchanged, "the deep-skips key is unchanged");
        let again = program_key(&build, false, true, false, false, false, None, &[]);
        assert_eq!(again, opt, "the same decision, the same key");
        let other = program_key(&[8; 32], false, true, false, false, false, None, &[]);
        assert_ne!(other, opt, "and still the build's");
    }

    #[test]
    fn a_program_that_takes_mask_copies_has_a_key_of_its_own() {
        let build = [7u8; 32];
        let copies = program_key(&build, false, false, true, false, false, None, &[]);
        assert_ne!(copies, build, "the build's key is for the programs that do not");
        assert_eq!(program_key(&build, false, false, true, false, false, None, &[]), copies, "the same decision, the same key");
        assert_ne!(program_key(&[8; 32], false, false, true, false, false, None, &[]), copies, "and still the build's");
        // Each combination a key of its own; the earlier ones unchanged.
        let all: Vec<[u8; 32]> = (0..8u8)
            .map(|n| program_key(&build, n & 1 != 0, n & 2 != 0, n & 4 != 0, false, false, None, &[]))
            .collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b);
            }
        }
        let mut h = blake3::Hasher::new();
        h.update(&build);
        h.update(b"VICTORY_WAVE_DEEP_SKIPS off for this program");
        assert_eq!(program_key(&build, true, false, false, false, false, None, &[]), *h.finalize().as_bytes());
    }

    #[test]
    fn a_program_with_dropped_stores_has_a_key_per_list_of_stores() {
        let build = [7u8; 32];
        let one = program_key(&build, false, false, false, false, false, None, &[0x2957c]);
        let two = program_key(&build, false, false, false, false, false, None, &[0x294b8, 0x29598]);
        assert_ne!(one, build);
        assert_ne!(one, two);
        assert_eq!(program_key(&build, false, false, false, false, false, None, &[0x2957c]), one, "the same stores, the same key");
        assert_ne!(program_key(&build, false, false, true, false, false, None, &[0x2957c]), one, "with the other decisions too");
        assert_eq!(program_key(&build, false, false, false, false, false, None, &[]), build, "none: the build's key");
    }

    #[test]
    fn a_program_built_with_site_pairs_has_a_key_of_its_own() {
        let build = [7u8; 32];
        let pairs = program_key(&build, false, false, false, true, false, None, &[]);
        let copies = program_key(&build, false, false, true, false, false, None, &[]);
        assert_ne!(pairs, build);
        assert_ne!(pairs, copies, "not the mask copies' key");
        assert_ne!(program_key(&build, false, false, true, true, false, None, &[]), pairs);
        assert_eq!(program_key(&build, false, false, false, true, false, None, &[]), pairs, "the same decision, the same key");
    }

    #[test]
    fn a_program_in_the_paired_form_has_a_key_per_setting() {
        let build = [7u8; 32];
        let fp32 = "VICTORY_WAVE_PAIRS_FP32=non-ieee-rne-ftz;VICTORY_LDS_INIT_HOIST=";
        let paired = program_key(&build, false, false, false, false, false, Some(fp32), &[]);
        let plain = program_key(&build, false, false, false, false, false, Some("VICTORY_WAVE_PAIRS_FP32=;VICTORY_LDS_INIT_HOIST="), &[]);
        assert_ne!(paired, build);
        assert_ne!(paired, plain, "another setting, another key");
        assert_ne!(paired, program_key(&build, false, false, false, true, false, None, &[]), "not the site pairs' key");
        assert_eq!(program_key(&build, false, false, false, false, false, Some(fp32), &[]), paired, "the same decision, the same key");
        let flags = program_key(&build, false, false, false, false, true, None, &[]);
        assert_ne!(flags, build);
        assert_ne!(flags, paired, "not the paired form's key");
        assert_ne!(flags, program_key(&build, false, false, false, true, false, None, &[]), "not the site pairs' key");
    }

    #[test]
    fn a_second_attempt_with_mask_copies_is_filed_apart_from_the_first() {
        use victory_shader::emit::{MASK_COPIES_RETRY_ON, MaskCopies};
        let build = [7u8; 32];
        let dir = tempfile::tempdir().unwrap();
        let cache = Cache::open(dir.path()).unwrap();
        // A program refused without the copies and translated with them, as #f133e35e.
        let runs = AtomicU64::new(0);
        let wave64 = || true;
        let attempt = |copies: bool| {
            let translate = |_: &[u8]| {
                runs.fetch_add(1, Ordering::Relaxed);
                if copies {
                    Ok(shader())
                } else {
                    Err(format!("GFX10 code at 0x1eb4: {MASK_COPIES_RETRY_ON} is not supported"))
                }
            };
            cached(&cache, &program_key(&build, false, false, copies, false, false, None, &[]), b"program", translate)
        };
        let (r, again) = attempts(MaskCopies::Retry, wave64, attempt);
        assert_eq!((r, again), (Ok(shader()), true));
        assert_eq!(runs.load(Ordering::Relaxed), 2);
        // The refusal is not cached; the second attempt is, under the copies' key, where
        // `all` finds it, and not under the first attempt's, where `0` looks.
        let (r, again) = attempts(MaskCopies::Retry, wave64, attempt);
        assert_eq!((r, again), (Ok(shader()), true));
        assert_eq!(runs.load(Ordering::Relaxed), 3, "the first attempt again, the second from the cache");
        assert_eq!(attempts(MaskCopies::On, wave64, attempt).0, Ok(shader()));
        assert_eq!(runs.load(Ordering::Relaxed), 3, "all: from the cache");
        assert!(attempts(MaskCopies::Off, wave64, attempt).0.is_err(), "0: refused, as before");
        // Not for a program that is no console wave64 one, nor after another refusal.
        assert!(attempts(MaskCopies::Retry, || false, attempt).0.is_err());
        let other = |copies: bool| if copies { Ok(1) } else { Err("an opaque lane mask".to_owned()) };
        assert_eq!(attempts(MaskCopies::Retry, wave64, other), (Err("an opaque lane mask".to_owned()), false));
        // A program that translates without the copies never takes them under `retry`.
        let fine = |copies: bool| Ok::<_, String>(copies);
        assert_eq!(attempts(MaskCopies::Retry, wave64, fine), (Ok(false), false));
        assert_eq!(attempts(MaskCopies::On, wave64, fine), (Ok(true), false));
        assert_eq!(attempts(MaskCopies::On, || false, fine), (Ok(false), false));
        // Refused again with the copies: the first refusal.
        let both = |copies: bool| Err::<u8, _>(format!("{MASK_COPIES_RETRY_ON} ({copies})"));
        assert_eq!(attempts(MaskCopies::Retry, wave64, both), (Err(format!("{MASK_COPIES_RETRY_ON} (false)")), true));
    }

    #[test]
    fn entries_round_trip_and_reject_damage() {
        let bytes = encode(&shader());
        assert_eq!(decode(&bytes), Some(shader()));
        assert_eq!(decode(&bytes[..bytes.len() - 1]), None);
        assert_eq!(decode(&[bytes.as_slice(), &[0]].concat()), None);
        assert_eq!(decode(b"VSH0"), None);
    }
}
