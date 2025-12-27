use serial_test::serial;
use xgrammar::Grammar;

#[test]
#[serial]
fn test_from_json_schema_returns_err_instead_of_aborting() {
    // This schema is invalid because it disallows additional items (items=false),
    // provides no prefixItems, yet requires minItems > prefixItems.len().
    let schema =
        r#"{"type":"array","prefixItems":[],"items":false,"minItems":2}"#;

    let err = Grammar::from_json_schema(
        schema,
        true,
        None,
        None::<(&str, &str)>,
        true,
        None,
        false,
    )
    .err()
    .expect("expected from_json_schema to return Err for an invalid schema");

    // Message comes from the underlying C++ exception (xgrammar::LogFatalError).
    assert!(
        err.contains("minItems") || err.contains("prefixItems"),
        "unexpected error message: {err}"
    );
}
