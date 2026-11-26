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
pub(super) const PER_PROGRAM_SWITCHES: [&str; 5] = [
    "VICTORY_WAVE_DEEP_SKIPS_NOT",
    "VICTORY_WAVE_DEEP_SKIPS_ONLY",
    "VICTORY_SHADER_OPT_NOT",
    "VICTORY_SHADER_OPT_ONLY",
    "VICTORY_WAVE_MASK_COPIES",
];

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
}
