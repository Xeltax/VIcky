//! The host list's side of the texture copy-back ([`crate::texback`]): the
//! textures of the kinds copied back that a list's shaders write are noted
//! as the draw or dispatch is prepared, and copied into read-back buffers
//! when the list closes; their texels land in title memory once the list has
//! run (`read_backs`). After fxceless/victory-fork #999 (`f5c03f3b0f`) and
//! #1376 (`86d446b74f`); a module of its own to keep `host.rs` small.

use super::HostList;

impl HostList {
    /// A draw or dispatch being prepared writes title texture `resource`
    /// through unordered access: if it is of a kind copied back
    /// ([`crate::texback`]), the list copies it when it closes.
    pub(crate) fn texture_written(&self, resource: crate::object::This) {
        if !crate::texback::wanted(resource) {
            return;
        }
        let mut deferred = self.deferred();
        // A target whose ticket went stale (the title reused or remapped the
        // memory it tiles into) must be re-registered, or the read-back
        // would be dropped as one of memory the texture no longer holds.
        let existing = deferred
            .back_wanted
            .iter_mut()
            .find(|target| target.owner() == resource as usize);
        match existing {
            Some(entry) if !entry.fresh() => {
                if let Some(target) = crate::texback::Target::new(resource) {
                    *entry = target;
                }
            }
            None => deferred.back_wanted.extend(crate::texback::Target::new(resource)),
            Some(_) => {}
        }
    }

    /// Before the list closes: the textures its shaders wrote that are copied
    /// back go into read-back buffers ([`crate::texback::record`]).
    pub(super) fn textures_back(&self) {
        const COPY_SOURCE: u32 = 0x800;
        let wanted = std::mem::take(&mut self.deferred().back_wanted);
        for target in wanted {
            self.settle(&target.copy, COPY_SOURCE);
            if let Some(back) = crate::texback::record(&self.list, target) {
                self.deferred().texture_backs.push(back);
            }
        }
    }

    /// For tests: whether the list's read-back wants a *fresh* target for
    /// `resource` — one whose ticket has not gone stale. A stale target is
    /// one the title's memory reuse invalidated, which the list must
    /// re-register on the next write of `resource`.
    #[cfg(test)]
    pub(crate) fn back_wanted_fresh(&self, resource: crate::object::This) -> bool {
        self.deferred()
            .back_wanted
            .iter()
            .any(|t| t.owner() == resource as usize && t.fresh())
    }

    /// For tests: how many texture read-backs the list recorded at its close.
    #[cfg(test)]
    pub(crate) fn texture_back_count(&self) -> usize {
        self.deferred().texture_backs.len()
    }
}
