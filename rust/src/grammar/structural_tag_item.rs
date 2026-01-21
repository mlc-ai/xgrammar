/// Deprecated. Definition of a structural tag item.
///
/// See :meth:`xgrammar.Grammar.from_structural_tag` for more details.
#[derive(Debug, Clone)]
pub struct StructuralTagItem {
    /// The begin tag.
    pub begin: String,
    /// The schema.
    pub schema: String,
    /// The end tag.
    pub end: String,
}

impl StructuralTagItem {
    pub fn new(
        begin: impl Into<String>,
        schema: impl Into<String>,
        end: impl Into<String>,
    ) -> Self {
        Self {
            begin: begin.into(),
            schema: schema.into(),
            end: end.into(),
        }
    }
}
