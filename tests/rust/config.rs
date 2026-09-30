mod common;

#[test]
fn recursion_depth_and_version() {
    common::init();

    let initial = xgrammar::get_max_recursion_depth().unwrap();
    assert!(initial > 0);

    xgrammar::set_max_recursion_depth(1234).unwrap();
    assert_eq!(xgrammar::get_max_recursion_depth().unwrap(), 1234);

    {
        let _guard = xgrammar::max_recursion_depth(99).unwrap();
        assert_eq!(xgrammar::get_max_recursion_depth().unwrap(), 99);
    }
    assert_eq!(xgrammar::get_max_recursion_depth().unwrap(), 1234);
    xgrammar::set_max_recursion_depth(initial).unwrap();

    let version = xgrammar::get_serialization_version().unwrap();
    assert!(!version.is_empty());
}
