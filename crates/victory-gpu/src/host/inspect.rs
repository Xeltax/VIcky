//! VIcky: the host list's side of `VICTORY_BVH_INPUT_CHECK=1`
//! (diagnostics, `crate::draw`'s `bvh_input`): a read-only step in the
//! list's submission-time steps, and a question about the refreshes the
//! list has queued. Nothing here writes a copy or title memory.

use std::sync::Arc;

use super::{HostList, Step};
use crate::shadow::Shadow;

impl HostList {
    /// Whether a refresh queued in this list covers `[address, address +
    /// len)` of `shadow` (the refresh of a root view is queued when the
    /// view is bound, so before the dispatch is recorded).
    pub(crate) fn refresh_queued(&self, shadow: &Arc<Shadow>, address: usize, len: usize) -> bool {
        let ptr = Arc::as_ptr(shadow) as usize;
        self.deferred().copies.iter().any(|step| {
            matches!(step, Step::Refresh(s, a, l)
                if Arc::as_ptr(s) as usize == ptr && *a <= address && address + len <= a + l)
        })
    }

    /// Runs `work` when the list is submitted, in order with its steps and
    /// after `shadow`'s queued refreshes are done.
    pub(crate) fn defer_inspect(&self, shadow: &Arc<Shadow>, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Inspect(shadow.clone(), Box::new(work)));
    }
}
