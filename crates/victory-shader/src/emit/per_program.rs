//! Switches that pick a translation per program, by the hash a run's log
//! names the shader by: the first 16 hex digits of the BLAKE3 of the
//! container as the title passes it (`hash=` in "shader translated" and
//! "shader does not translate", `#hash` in `victory translate --list`).
//!
//! The one so far is for the wave64 form "without skips, scalars followed"
//! ([`crate::ir::without_skips`] with `deep`), the exact form tried for a
//! console wave64 compute program on a host of 32-lane waves before the
//! approximate per-half-wave ones. On an RTX 4080 (2026-11-24) it lost
//! walls, roofs and floors (about 5,000 draws a frame); off for every
//! program (`VICTORY_WAVE_DEEP_SKIPS=0`) those came back (about 6,900) and
//! motion looked wrong. So some of the programs that take it are wrong in
//! it and others better: it is chosen per program.
//! - `VICTORY_WAVE_DEEP_SKIPS=0`: off for every program, as before;
//! - `VICTORY_WAVE_DEEP_SKIPS_NOT=<hash>,...`: off for the listed programs;
//! - `VICTORY_WAVE_DEEP_SKIPS_ONLY=<hash>,...`: off for every program not
//!   listed.
//!
//! The other is for the clean-up passes over the finished module
//! (`VICTORY_SHADER_OPT`, [`super::shader_opt`]): on an RTX 4080 and an
//! AMD RX 7800 XT (2026-11-24) runs with them lost walls and body parts
//! that runs without them had, and the offline check (`bitcode::check`)
//! found no program they change. So a run can keep chosen programs out of
//! them, to find one:
//! - `VICTORY_SHADER_OPT_NOT=<hash>,...`: no clean-up for the listed
//!   programs;
//! - `VICTORY_SHADER_OPT_ONLY=<hash>,...`: no clean-up for every program not
//!   listed.
//!
//! The third is for a wave64 lane mask the compiler parked as two scalar
//! copies and restored dword by dword (`s_mov_b32` chains): structurisation
//! can take the restored pair as the mask it was copied from
//! (`structure`'s `Lanes::mask`, DexNom's 76749ce). On an RTX 4080
//! (2026-11-26, RT88a) that let `CS_ShadowRenderTiledCascadesAlphaCBPPPC`
//! (#f133e35e) translate and took the 8x8 stair-steps off shadow edges, and
//! the characters' skin was no longer drawn. Offline, over every program of
//! the title, it takes effect in five wave64 programs: two that were
//! refused translate (#f133e35e, and `PS_VoxelizeVisiblity` #405bfb31,
//! refused as "an opaque lane mask"), one that translated comes out
//! different (`water_shading:CS_TransmissionVolumetrics_VolumePortalsOnly`
//! #a10179ab), and two stay refused as before. Of these RT88a's scene bound
//! #f133e35e alone: the lost heads go with running it (runs that drop it
//! draw them; a tree whose own translator runs it loses them too), which
//! the default keeps and `0` undoes. So it is chosen per program:
//! - `VICTORY_WAVE_MASK_COPIES` unset or `retry`: only as a second attempt,
//!   for a program whose translation without it is refused where a mask is
//!   exchanged ([`MASK_COPIES_RETRY_ON`]): no program that translated
//!   before changes;
//! - `VICTORY_WAVE_MASK_COPIES=0`: for no program, as before;
//! - `VICTORY_WAVE_MASK_COPIES=all`: for every program, from the start;
//! - `VICTORY_WAVE_MASK_COPIES=<hash>,...`: for the listed programs, from
//!   the start; for no other.
//!
//! A program's decision is part of its shader cache key (`victory-gpu`'s
//! `shader_cache`), not the lists: changing a list translates again only
//! the programs whose decision it changes. For the mask copies the key says
//! whether the translation ran with them, so a second attempt is filed
//! where `all` files that program, never where its first attempt would be.
//!
//! And one to bisect what a program's writes do (diagnostics): its store
//! instructions at the listed console offsets are left out of its
//! translation, as if what they write went nowhere (a store to a null
//! view):
//! - `VICTORY_STORE_DROP=<hash>:<offset>+<offset>...,<hash>:...` (offsets
//!   in hex, with or without `0x`, as `victory shader-info --disasm` lists
//!   them; `buffer_`, `image_`, `global_` and `scratch_` stores, not
//!   atomics or exports). Run 2026-11-26: the stores of
//!   `CS_ShadowRenderTiledCascadesAlphaCBPPPC` (#f133e35e), to find which of
//!   its outputs costs the characters their skin.
//!
//! The same machinery leaves out, by default, the parent-link stores of the
//! ray-tracing structure builder's two box rehydration programs
//! ([`BVH_LINK_STORES`]); `VICTORY_BVH_LINK_STORES=keep` keeps them. The
//! offsets `VICTORY_STORE_DROP` lists are left out as well.

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::dxbc::Container;
use crate::native::HwStage;

thread_local! {
    /// The hash of the container being translated on this thread ([`for_program`]).
    static PROGRAM: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Runs `f`, the translation of the container whose hash is `hash` (see the
/// module notes): the per-program switches and `VICTORY_WAVE_WHY`'s
/// `WAVE64-RETRY` lines see it.
pub fn for_program<T>(hash: &str, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PROGRAM.set(self.0.take());
        }
    }
    let _restore = Restore(PROGRAM.replace(Some(hash.to_owned())));
    f()
}

/// The hash [`for_program`] set on this thread, if any.
pub(crate) fn current() -> Option<String> {
    PROGRAM.with_borrow(Clone::clone)
}

/// The switches keyed per program (the shader cache mixes each program's
/// decision into its own key) and so left out of the translator's
/// fingerprint ([`super::translation_fingerprint`]): a new list translates
/// again only the programs whose decision it changes.
pub(super) const PER_PROGRAM_SWITCHES: [&str; 8] = [
    "VICTORY_WAVE_DEEP_SKIPS_NOT",
    "VICTORY_WAVE_DEEP_SKIPS_ONLY",
    "VICTORY_SHADER_OPT_NOT",
    "VICTORY_SHADER_OPT_ONLY",
    "VICTORY_WAVE_MASK_COPIES",
    "VICTORY_STORE_DROP",
    "VICTORY_WAVE_SITE_PAIRS",
    "VICTORY_BVH_LINK_STORES",
];

/// The parent-link stores of the ray-tracing structure builder's box rehydration programs,
/// 16-bit boxes (#caa78b8c, 64-byte nodes) and 32-bit boxes (#28918b71, 128-byte nodes),
/// left out of their translations unless `VICTORY_BVH_LINK_STORES=keep`.
///
/// Each thread writes one node in place over the compressed data, then for each of its four
/// children whose type takes one, one dword: its node's address at
/// `c1 + ((((child << 3) & !63) - c2) >> 4 & 0x0ffffffc)`, an offset the child pointer
/// alone decides. With ray tracing on (`VICTORY_BVH_BUILDER=1`, RTX 4080, 32-lane waves) a
/// save loaded into the first chapter made the title read all ones at `+0x108557`, an
/// object a few KiB past a rehydrated structure that a rehydration dispatch had written
/// (fxceless/victory-fork#16, follow-ups 13 and 14). Runs on that save (2026-11-27):
/// - the four rehydration stages skipped (`VICTORY_BVH_SKIP=47-50`): no crash;
/// - only these two programs skipped (`49-50`): no crash, with every copy number used up
///   in the builder's first frame as in the crashing run;
/// - all four stages run, these 16 stores left out (`VICTORY_STORE_DROP`): no crash.
///
/// So some child pointer these stores read holds something else (f16 bounds, another
/// node's words) and sends the dword out of the structure. The translations were checked
/// (2026-11-27): read through (`dxc -dumpbin`), and run in place on the RTX 4080 over
/// synthetic compressed data (512 groups, last groups of 1 to 64 threads, 0 to 4 children of
/// each kind), where every node and every parent link came out as the console's code writes
/// them. So the pointers are most likely wrong in the data the programs are given, not
/// computed wrong; where from is not known yet (VICKY-RT.md). What is lost: the parent links
/// of rehydrated structures, which a refit walks up; traversal does not read them.
pub(super) const BVH_LINK_STORES: [(&str, [usize; 8]); 2] = [
    ("caa78b8ca62bc3d6", [0xb44, 0xb90, 0xbcc, 0xc18, 0xc54, 0xca0, 0xcdc, 0xd28]),
    ("28918b7150a7aad5", [0x904, 0x950, 0x98c, 0x9d8, 0xa14, 0xa60, 0xa9c, 0xae8]),
];

/// Whether the program whose hash is `hash` (`None`: not known) is built with fork #1002's
/// forms (brunofgmag, `eaeb8f17`; `crate::ir`): a pair of scalars parked in two lanes that
/// hold other scalars elsewhere moves as a pair where it can (`site_pairs`), an
/// `s_cbranch_execnz` over a far jump to the end is an early-out, and a scalar join that
/// passes one value through is a copy of it (`without_skips` with `deep`). With them the
/// tiled cascade shadow programs (#f133e35e and three siblings) translate exactly on a host
/// of 32-lane waves, without the mask copies. `VICTORY_WAVE_SITE_PAIRS` (opt-in): unset or
/// `0` for no program, `all` for every one, `<hash>,...` for the listed ones.
pub fn site_pairs_allowed(hash: Option<&str>) -> bool {
    #[cfg(test)]
    if let Some(value) = TEST_SITE_PAIRS.get() {
        return site_pairs_decide(value, hash);
    }
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    let value = VALUE.get_or_init(|| std::env::var(PER_PROGRAM_SWITCHES[6]).ok());
    site_pairs_decide(value.as_deref(), hash)
}

/// [`site_pairs_allowed`] for the program being translated on this thread ([`for_program`]).
pub(crate) fn site_pairs_here() -> bool {
    site_pairs_allowed(current().as_deref())
}

#[cfg(test)]
thread_local! {
    /// A test's value of `VICTORY_WAVE_SITE_PAIRS` (`Some(None)`: unset), for its thread only.
    pub(crate) static TEST_SITE_PAIRS: std::cell::Cell<Option<Option<&'static str>>> =
        const { std::cell::Cell::new(None) };
}

/// [`site_pairs_allowed`] on the switch's value (`None`: unset). Pure, for testing.
fn site_pairs_decide(value: Option<&str>, hash: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("0") => false,
        Some(v) if v.eq_ignore_ascii_case("all") => true,
        Some(v) => listed(&parse_hash_list(v), hash),
    }
}

/// The console offsets of the stores left out of the translation of the
/// program whose hash is `hash` (`None`: not known), in increasing order:
/// those `VICTORY_STORE_DROP` lists (module notes) and [`BVH_LINK_STORES`];
/// empty for most.
pub fn dropped_stores(hash: Option<&str>) -> Vec<usize> {
    #[cfg(test)]
    if TEST_STORE_DROP.get().is_some() || TEST_BVH_LINK_STORES.get().is_some() {
        let list = drop_list(TEST_STORE_DROP.get().unwrap_or_default(), TEST_BVH_LINK_STORES.get().flatten());
        return stores_for(&list, hash);
    }
    static LIST: OnceLock<Vec<(String, Vec<usize>)>> = OnceLock::new();
    let list = LIST.get_or_init(|| {
        let links = std::env::var(PER_PROGRAM_SWITCHES[7]).ok();
        if !links_kept(links.as_deref()) {
            eprintln!(
                "VICTORY_BVH_LINK_STORES: the parent-link stores of #caa78b8c and #28918b71 are left out (`keep` keeps them)"
            );
        }
        drop_list(&std::env::var(PER_PROGRAM_SWITCHES[5]).unwrap_or_default(), links.as_deref())
    });
    stores_for(list, hash)
}

/// Whether the store at console offset `offset` of the program being
/// translated on this thread ([`for_program`]) is left out
/// (`VICTORY_STORE_DROP`).
pub(crate) fn store_dropped_here(offset: usize) -> bool {
    dropped_stores(current().as_deref()).contains(&offset)
}

#[cfg(test)]
thread_local! {
    /// A test's value of `VICTORY_STORE_DROP`, for its thread only.
    pub(crate) static TEST_STORE_DROP: std::cell::Cell<Option<&'static str>> = const { std::cell::Cell::new(None) };
    /// A test's value of `VICTORY_BVH_LINK_STORES` (`Some(None)`: unset), for its thread only.
    pub(crate) static TEST_BVH_LINK_STORES: std::cell::Cell<Option<Option<&'static str>>> =
        const { std::cell::Cell::new(None) };
}

/// Whether `VICTORY_BVH_LINK_STORES`'s value (`None`: unset) keeps the stores of
/// [`BVH_LINK_STORES`]: only `keep` does.
fn links_kept(value: Option<&str>) -> bool {
    value.is_some_and(|v| v.trim().eq_ignore_ascii_case("keep"))
}

/// The stores left out: `VICTORY_STORE_DROP`'s entries (`store_drop`), then
/// [`BVH_LINK_STORES`] unless `links` is `keep`. Pure, for testing.
fn drop_list(store_drop: &str, links: Option<&str>) -> Vec<(String, Vec<usize>)> {
    let mut list = parse_store_drop(store_drop);
    if !links_kept(links) {
        list.extend(BVH_LINK_STORES.iter().map(|(hash, offsets)| ((*hash).to_owned(), offsets.to_vec())));
    }
    list
}

/// `VICTORY_STORE_DROP`'s entries: (hash or its first digits, offsets).
/// An entry it cannot read is left out with a warning. Pure, for testing.
fn parse_store_drop(value: &str) -> Vec<(String, Vec<usize>)> {
    value
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .filter(|entry| !entry.is_empty())
        .filter_map(|entry| {
            let read = || -> Option<(String, Vec<usize>)> {
                let (hash, offsets) = entry.split_once(':')?;
                let hash = hash.trim_start_matches('#').to_ascii_lowercase();
                if hash.is_empty() || hash.len() > 16 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return None;
                }
                let offsets = offsets
                    .split('+')
                    .map(|o| {
                        let o = o.trim();
                        let o = o.strip_prefix("0x").or_else(|| o.strip_prefix("0X")).unwrap_or(o);
                        usize::from_str_radix(o, 16).ok()
                    })
                    .collect::<Option<Vec<usize>>>()?;
                (!offsets.is_empty()).then_some((hash, offsets))
            };
            let parsed = read();
            if parsed.is_none() {
                eprintln!("VICTORY_STORE_DROP: an entry it cannot read, left out: {entry} (<hash>:<offset>+<offset>, in hex)");
            }
            parsed
        })
        .collect()
}

/// The offsets `list` names for the program `hash`, sorted and each once.
fn stores_for(list: &[(String, Vec<usize>)], hash: Option<&str>) -> Vec<usize> {
    let mut out: Vec<usize> = list
        .iter()
        .filter(|(entry, _)| listed(std::slice::from_ref(entry), hash))
        .flat_map(|(_, offsets)| offsets.iter().copied())
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// What `VICTORY_WAVE_MASK_COPIES` asks of a program's translation (module
/// notes): whether a wave64 lane mask restored from scalar copies is taken
/// as the mask it was copied from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskCopies {
    /// Never (`0`; a list that does not name the program).
    Off,
    /// From the start (`all`; a list that names the program).
    On,
    /// Only as a second attempt after a first without them was refused with
    /// [`MASK_COPIES_RETRY_ON`] (unset, `retry`).
    Retry,
}

/// The refusal that gives a program a second attempt with the mask copies
/// ([`MaskCopies::Retry`]): the one #f133e35e has without them.
pub const MASK_COPIES_RETRY_ON: &str = super::lower::MASK_EXCHANGED;

thread_local! {
    /// Whether the translation on this thread takes the mask copies whatever the switch says
    /// ([`with_mask_copies`]).
    static COPIES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
thread_local! {
    /// A test's value of `VICTORY_WAVE_MASK_COPIES` (`Some(None)`: unset), for its thread
    /// only.
    pub(crate) static TEST_MASK_COPIES: std::cell::Cell<Option<Option<&'static str>>> =
        const { std::cell::Cell::new(None) };
}

/// Whether the deep-skips form may be tried for the program whose hash is
/// `hash` (`None`: not known, as for a translation outside
/// [`for_program`]), by the switches in the module notes.
pub fn deep_skips_allowed(hash: Option<&str>) -> bool {
    struct Switches {
        on: bool,
        not: Option<Vec<String>>,
        only: Option<Vec<String>>,
    }
    static SWITCHES: OnceLock<Switches> = OnceLock::new();
    let s = SWITCHES.get_or_init(|| {
        let list = |name| std::env::var(name).ok().map(|v| parse_hash_list(&v));
        Switches {
            on: std::env::var_os("VICTORY_WAVE_DEEP_SKIPS").is_none_or(|v| v != "0"),
            not: list(PER_PROGRAM_SWITCHES[0]),
            only: list(PER_PROGRAM_SWITCHES[1]),
        }
    });
    decide(s.on, s.not.as_deref(), s.only.as_deref(), hash)
}

/// Whether the clean-up passes (`VICTORY_SHADER_OPT`) run over the program
/// whose hash is `hash` (`None`: not known, as for a translation outside
/// [`for_program`]), by `VICTORY_SHADER_OPT_NOT` and `_ONLY` (module notes).
/// Which passes run is [`super::shader_opt`]; with none, the lists do not
/// matter.
pub fn shader_opt_allowed(hash: Option<&str>) -> bool {
    struct Lists {
        not: Option<Vec<String>>,
        only: Option<Vec<String>>,
    }
    static LISTS: OnceLock<Lists> = OnceLock::new();
    let l = LISTS.get_or_init(|| {
        let list = |name| std::env::var(name).ok().map(|v| parse_hash_list(&v));
        Lists {
            not: list(PER_PROGRAM_SWITCHES[2]),
            only: list(PER_PROGRAM_SWITCHES[3]),
        }
    });
    decide(true, l.not.as_deref(), l.only.as_deref(), hash)
}

/// What `VICTORY_WAVE_MASK_COPIES` (module notes) asks of the translation of
/// the program whose hash is `hash` (`None`: not known, as for a translation
/// outside [`for_program`]).
pub fn mask_copies(hash: Option<&str>) -> MaskCopies {
    #[cfg(test)]
    if let Some(value) = TEST_MASK_COPIES.get() {
        return mask_copies_decide(value, hash);
    }
    static VALUE: OnceLock<Option<String>> = OnceLock::new();
    let value = VALUE.get_or_init(|| std::env::var(PER_PROGRAM_SWITCHES[4]).ok());
    mask_copies_decide(value.as_deref(), hash)
}

/// Runs `f` with the mask copies taken (`on`) or not, whatever
/// `VICTORY_WAVE_MASK_COPIES` says: for a caller that makes each attempt
/// itself and files it under its own key (`victory-gpu`'s `shader_cache`).
/// [`super::emit`] then makes no second attempt of its own.
pub fn with_mask_copies<T>(on: bool, f: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            COPIES.set(self.0);
        }
    }
    let _restore = Restore(COPIES.replace(Some(on)));
    f()
}

/// Whether the translation on this thread takes the mask copies: as
/// [`with_mask_copies`] says, else as the switch says for the program
/// ([`for_program`]; a second attempt runs inside [`with_mask_copies`]).
pub(crate) fn mask_copies_here() -> bool {
    COPIES.get().unwrap_or_else(|| mask_copies(current().as_deref()) == MaskCopies::On)
}

/// Whether a translation on this thread that was refused with `what` is
/// attempted once more with the mask copies: the switch says
/// [`MaskCopies::Retry`] for the program and no caller chose
/// ([`with_mask_copies`]).
pub(crate) fn mask_copies_retry_here(what: &str) -> bool {
    COPIES.get().is_none()
        && mask_copies_retried(what)
        && mask_copies(current().as_deref()) == MaskCopies::Retry
}

/// Whether a translation refused with `error` (its text, or the `what` of a
/// [`crate::ShaderError::Unsupported`]) gets a second attempt with the mask
/// copies under [`MaskCopies::Retry`].
pub fn mask_copies_retried(error: &str) -> bool {
    error.contains(MASK_COPIES_RETRY_ON)
}

/// [`mask_copies`] on the switch's value (`None`: unset). Pure, for testing.
fn mask_copies_decide(value: Option<&str>, hash: Option<&str>) -> MaskCopies {
    let on = |yes: bool| if yes { MaskCopies::On } else { MaskCopies::Off };
    match value.map(str::trim) {
        None => MaskCopies::Retry,
        Some(v) if v.eq_ignore_ascii_case("retry") => MaskCopies::Retry,
        // (not the list of the programs whose hash begins with a 0)
        Some("0") => MaskCopies::Off,
        Some(v) if v.eq_ignore_ascii_case("all") => MaskCopies::On,
        Some(v) => on(listed(&parse_hash_list(v), hash)),
    }
}

/// Whether the program of `container` is a console wave64 program, the only
/// kind `VICTORY_WAVE_MASK_COPIES` can change. Cheap (no translation).
pub fn wave64_program(container: &[u8]) -> bool {
    Container::parse(container)
        .ok()
        .and_then(|c| super::main_program(&c).ok())
        .is_some_and(|p| p.wave64)
}

/// Whether the wave64 forms, and so the deep-skips switches, can matter for
/// the program of `container`: a console wave64 compute program, on a host
/// of 32-lane waves. Cheap (no translation): a superset of the programs
/// that take the deep-skips form.
pub fn wave64_retried(container: &[u8]) -> bool {
    super::lower::host_max_wave() == 32
        && Container::parse(container)
            .ok()
            .and_then(|c| super::main_program(&c).ok())
            .is_some_and(|p| p.stage == HwStage::Cs && p.wave64)
}

/// The hashes of a `_NOT` or `_ONLY` list: separated by commas, semicolons
/// or white space, each with or without the `#` the offline listing puts
/// before it, in lower case and cut to the 16 digits a hash has. Pure, for
/// testing.
fn parse_hash_list(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c == ';' || c.is_whitespace())
        .map(|h| h.trim_start_matches('#').to_ascii_lowercase())
        .map(|h| h.chars().take(16).collect::<String>())
        .filter(|h| !h.is_empty())
        .collect()
}

/// Whether `list` names the program `hash`: an entry names every program
/// whose hash begins with it.
fn listed(list: &[String], hash: Option<&str>) -> bool {
    hash.is_some_and(|hash| {
        let hash = hash.to_ascii_lowercase();
        list.iter().any(|entry| hash.starts_with(entry.as_str()))
    })
}

/// [`deep_skips_allowed`] on the switches' values. Pure, for testing.
fn decide(on: bool, not: Option<&[String]>, only: Option<&[String]>, hash: Option<&str>) -> bool {
    on && !not.is_some_and(|l| listed(l, hash)) && only.is_none_or(|l| listed(l, hash))
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "0123456789abcdef";
    const B: &str = "fedcba9876543210";

    fn list(v: &str) -> Vec<String> {
        parse_hash_list(v)
    }

    #[test]
    fn lists_split_on_commas_and_space_and_take_the_listing_form() {
        assert_eq!(list(""), Vec::<String>::new());
        assert_eq!(list(" , ;"), Vec::<String>::new());
        assert_eq!(list(A), vec![A]);
        assert_eq!(list(&format!("{A},{B}")), vec![A, B]);
        assert_eq!(list(&format!(" #{A} ;\t{B} ,")), vec![A, B]);
        assert_eq!(list("#0123456789ABCDEF"), vec![A], "upper case");
        // The whole BLAKE3 is cut to the hash a log shows.
        assert_eq!(list(&format!("{A}{B}{A}{B}")), vec![A]);
        // A shorter entry stays short: it names every hash that begins with it.
        assert_eq!(list("01234567"), vec!["01234567"]);
    }

    #[test]
    fn without_lists_the_global_switch_decides() {
        for hash in [Some(A), Some(B), None] {
            assert!(decide(true, None, None, hash));
            assert!(!decide(false, None, None, hash));
        }
    }

    #[test]
    fn not_keeps_the_listed_programs_out() {
        let not = list(&format!("#{B}, 01234567"));
        assert!(!decide(true, Some(&not), None, Some(A)), "by its first digits");
        assert!(!decide(true, Some(&not), None, Some(B)));
        assert!(!decide(true, Some(&not), None, Some(&B.to_ascii_uppercase())));
        assert!(!decide(true, Some(&not), None, Some("0123456700000000")));
        assert!(decide(true, Some(&not), None, Some("1111111111111111")));
        assert!(decide(true, Some(&not), None, None), "an unknown program is not listed");
        assert!(decide(true, Some(&[]), None, Some(A)), "an empty list keeps everything");
        assert!(!decide(false, Some(&not), None, Some("1111111111111111")), "=0 still wins");
    }

    #[test]
    fn only_lets_just_the_listed_programs_in() {
        let only = list(A);
        assert!(decide(true, None, Some(&only), Some(A)));
        assert!(!decide(true, None, Some(&only), Some(B)));
        assert!(!decide(true, None, Some(&only), None), "an unknown program is not listed");
        assert!(!decide(true, None, Some(&[]), Some(A)), "an empty list lets none in");
        assert!(!decide(false, None, Some(&only), Some(A)), "=0 still wins");
        // Both: listed in ONLY and not in NOT.
        let both = list(&format!("{A},{B}"));
        assert!(!decide(true, Some(&only), Some(&both), Some(A)));
        assert!(decide(true, Some(&only), Some(&both), Some(B)));
    }

    #[test]
    fn the_program_hash_is_seen_inside_its_translation_only() {
        assert_eq!(current(), None);
        let inner = for_program(A, || {
            let nested = for_program(B, current);
            (current(), nested)
        });
        assert_eq!(inner, (Some(A.to_owned()), Some(B.to_owned())));
        assert_eq!(current(), None);
        let _ = std::panic::catch_unwind(|| for_program(A, || panic!("a translation that fails")));
        assert_eq!(current(), None, "restored after a panic too");
    }

    #[test]
    fn what_is_no_container_is_not_retried() {
        for lanes in [0, 32, 64] {
            let was = super::super::lower::TEST_HOST_WAVE.replace(Some(lanes));
            assert!(!wave64_retried(b"not a container"), "{lanes}");
            assert!(!wave64_retried(b"DXBC"), "{lanes}");
            super::super::lower::TEST_HOST_WAVE.set(was);
        }
        assert!(!wave64_program(b"not a container"));
        assert!(!wave64_program(b"DXBC"));
    }

    #[test]
    fn mask_copies_go_by_the_switch_and_default_to_a_second_attempt() {
        use MaskCopies::{Off, On, Retry};
        const SHADOW: &str = "f133e35efed2a48a";
        for hash in [Some(A), Some(B), Some(SHADOW), None] {
            assert_eq!(mask_copies_decide(None, hash), Retry, "unset: {hash:?}");
            assert_eq!(mask_copies_decide(Some("retry"), hash), Retry, "retry: {hash:?}");
            assert_eq!(mask_copies_decide(Some(" RETRY "), hash), Retry, "retry: {hash:?}");
            assert_eq!(mask_copies_decide(Some("0"), hash), Off, "0: {hash:?}");
            assert_eq!(mask_copies_decide(Some(" 0 "), hash), Off, "0: {hash:?}");
            assert_eq!(mask_copies_decide(Some(""), hash), Off, "an empty list: {hash:?}");
            assert_eq!(mask_copies_decide(Some("all"), hash), On, "all: {hash:?}");
            assert_eq!(mask_copies_decide(Some("ALL"), hash), On, "all: {hash:?}");
        }
        let list = Some("#0123456789abcdef, fedcba98");
        assert_eq!(mask_copies_decide(list, Some(A)), On);
        assert_eq!(mask_copies_decide(list, Some(B)), On, "by its first digits");
        assert_eq!(mask_copies_decide(list, Some(SHADOW)), Off, "a list: no retry for the others");
        assert_eq!(mask_copies_decide(list, None), Off);
        // The refusal the retry is for, as `what` and as a whole error's text.
        assert!(mask_copies_retried(MASK_COPIES_RETRY_ON));
        let text = crate::ShaderError::Unsupported { offset: 0x1eb4, what: MASK_COPIES_RETRY_ON }.to_string();
        assert!(mask_copies_retried(&text), "{text}");
        assert!(!mask_copies_retried("a lane mask holding lanes elsewhere read as data"));
        assert!(!mask_copies_retried("an opaque lane mask"));
        // The test override is this thread's alone and takes the switch's place.
        let was = TEST_MASK_COPIES.replace(Some(Some("all")));
        assert_eq!(mask_copies(Some(A)), On);
        assert!(for_program(A, mask_copies_here));
        assert!(!for_program(A, || mask_copies_retry_here(MASK_COPIES_RETRY_ON)), "all: no second attempt");
        TEST_MASK_COPIES.set(Some(None));
        assert_eq!(mask_copies(Some(SHADOW)), Retry);
        // retry: the first attempt without; a second one after the refusal, unless the caller
        // chose ([`with_mask_copies`]).
        assert!(!for_program(SHADOW, mask_copies_here));
        assert!(for_program(SHADOW, || mask_copies_retry_here(MASK_COPIES_RETRY_ON)));
        assert!(!for_program(SHADOW, || mask_copies_retry_here("an opaque lane mask")));
        assert!(with_mask_copies(true, mask_copies_here));
        assert!(!with_mask_copies(false, || mask_copies_retry_here(MASK_COPIES_RETRY_ON)));
        assert!(!with_mask_copies(true, || with_mask_copies(false, mask_copies_here)));
        assert!(with_mask_copies(true, || {
            let _ = with_mask_copies(false, mask_copies_here);
            mask_copies_here()
        }), "restored after the inner one");
        assert!(!mask_copies_here(), "and after the outer one");
        TEST_MASK_COPIES.set(Some(Some("0")));
        assert!(!for_program(SHADOW, || mask_copies_retry_here(MASK_COPIES_RETRY_ON)));
        TEST_MASK_COPIES.set(was);
    }

    #[test]
    fn site_pairs_are_opt_in_per_program() {
        const SHADOW: &str = "f133e35efed2a48a";
        for hash in [Some(A), Some(SHADOW), None] {
            assert!(!site_pairs_decide(None, hash), "unset: {hash:?}");
            assert!(!site_pairs_decide(Some("0"), hash), "0: {hash:?}");
            assert!(site_pairs_decide(Some("all"), hash), "all: {hash:?}");
        }
        assert!(site_pairs_decide(Some("f133e35e"), Some(SHADOW)));
        assert!(!site_pairs_decide(Some("f133e35e"), Some(A)));
        let was = TEST_SITE_PAIRS.replace(Some(Some("f133e35e")));
        assert!(for_program(SHADOW, site_pairs_here));
        assert!(!for_program(A, site_pairs_here));
        assert!(!site_pairs_here(), "outside a program's translation");
        TEST_SITE_PAIRS.set(was);
    }

    #[test]
    fn dropped_stores_go_by_hash_and_offset() {
        const SHADOW: &str = "f133e35efed2a48a";
        let list = parse_store_drop("f133e35e:2957c+0x294b8, #0123456789abcdef:10 f133e35efed2a48a:2957C");
        assert_eq!(list.len(), 3);
        assert_eq!(stores_for(&list, Some(SHADOW)), vec![0x294b8, 0x2957c], "sorted, once each, by prefix");
        assert_eq!(stores_for(&list, Some(A)), vec![0x10]);
        assert_eq!(stores_for(&list, Some(B)), Vec::<usize>::new());
        assert_eq!(stores_for(&list, None), Vec::<usize>::new(), "an unknown program is not listed");
        // Entries it cannot read are left out, the others kept.
        let bad = parse_store_drop("f133e35e,f133e35e:,zz:10,f133e35e:12+xy,0123:20");
        assert_eq!(bad, vec![("0123".to_owned(), vec![0x20])]);
        assert!(parse_store_drop("").is_empty());
        // The test override, and the program on this thread.
        let was = TEST_STORE_DROP.replace(Some("f133e35e:2957c"));
        assert_eq!(dropped_stores(Some(SHADOW)), vec![0x2957c]);
        assert!(for_program(SHADOW, || store_dropped_here(0x2957c)));
        assert!(!for_program(SHADOW, || store_dropped_here(0x294b8)));
        assert!(!for_program(A, || store_dropped_here(0x2957c)));
        assert!(!store_dropped_here(0x2957c), "outside a program's translation");
        TEST_STORE_DROP.set(was);
    }

    #[test]
    fn the_builders_parent_link_stores_are_left_out_unless_kept() {
        const BOX16: &str = "caa78b8ca62bc3d6";
        const BOX32: &str = "28918b7150a7aad5";
        let box16 = vec![0xb44, 0xb90, 0xbcc, 0xc18, 0xc54, 0xca0, 0xcdc, 0xd28];
        let box32 = vec![0x904, 0x950, 0x98c, 0x9d8, 0xa14, 0xa60, 0xa9c, 0xae8];
        // unset, or any value but `keep`: left out, from those two programs only
        for links in [None, Some("0"), Some("1"), Some("")] {
            let list = drop_list("", links);
            assert_eq!(stores_for(&list, Some(BOX16)), box16, "{links:?}");
            assert_eq!(stores_for(&list, Some(BOX32)), box32, "{links:?}");
            assert_eq!(stores_for(&list, Some(A)), Vec::<usize>::new(), "{links:?}");
            assert_eq!(stores_for(&list, None), Vec::<usize>::new(), "{links:?}");
        }
        for links in ["keep", " KEEP "] {
            let list = drop_list("", Some(links));
            assert_eq!(stores_for(&list, Some(BOX16)), Vec::<usize>::new(), "{links:?}");
            assert_eq!(stores_for(&list, Some(BOX32)), Vec::<usize>::new(), "{links:?}");
        }
        // `VICTORY_STORE_DROP` adds to them, each offset once, or stands alone with `keep`
        let mut more = box16.clone();
        more.push(0xd30);
        assert_eq!(stores_for(&drop_list("caa78b8c:b44+d30", None), Some(BOX16)), more);
        assert_eq!(stores_for(&drop_list("caa78b8c:d30", Some("keep")), Some(BOX16)), vec![0xd30]);
        // The test overrides, and the program on this thread.
        let was = TEST_BVH_LINK_STORES.replace(Some(None));
        assert_eq!(dropped_stores(Some(BOX32)), box32);
        assert!(for_program(BOX16, || store_dropped_here(0xb44)));
        assert!(!for_program(BOX16, || store_dropped_here(0x904)));
        assert!(!for_program(A, || store_dropped_here(0xb44)));
        TEST_BVH_LINK_STORES.set(Some(Some("keep")));
        assert!(!for_program(BOX16, || store_dropped_here(0xb44)));
        assert!(dropped_stores(Some(BOX32)).is_empty());
        TEST_BVH_LINK_STORES.set(was);
    }
}
