# Godot Stat Math Testing Organization Policy

## Policy Statement
Effective immediately, all test suites in the Godot Stat Math project will be reorganized into dedicated folders containing separate files for each of the three standardized testing sections. This policy improves maintainability, readability, parallel development capabilities, and testing standards compliance.

## Folder Structure Pattern
```
tests/core/
├── [module_name]/
│   ├── scipy_validation_test.gd     # SCIPY VALIDATION TESTS - DATA-DRIVEN
│   ├── mathematical_property_test.gd # MATHEMATICAL PROPERTY TESTS  
│   └── parameter_validation_test.gd  # PARAMETER VALIDATION TESTS
```

## Implementation Guidelines

### File Naming Convention
- **scipy_validation_test.gd** - Contains all data-driven tests that validate against SciPy reference values
- **mathematical_property_test.gd** - Contains tests for mathematical properties, relationships, and theoretical behavior
- **parameter_validation_test.gd** - Contains all input validation and error handling tests

### Class Naming Convention
- **[ModuleName]ScipyValidationTest** extends GdUnitTestSuite
- **[ModuleName]MathematicalPropertyTest** extends GdUnitTestSuite  
- **[ModuleName]ParameterValidationTest** extends GdUnitTestSuite

### Section Headers
Each file must include the appropriate section header:
```gdscript
# =============================================================================
# [SECTION TYPE] TESTS
# =============================================================================
```

## Benefits

### 🎯 **Improved Maintainability**
- Smaller, focused files are easier to navigate and modify
- Changes to one testing aspect don't affect others
- Reduced merge conflicts in team development

### 🚀 **Enhanced Development Workflow**
- Parallel development on different test aspects
- Specialized testing focus areas
- Faster test execution when running specific test types

### 📊 **Better Organization**
- Clear separation of concerns
- Consistent structure across all modules
- Easier to identify missing test coverage

### 🔧 **Standards Compliance**
- Enforces the three-section testing standard
- Prevents mixing of test types within sections
- Maintains consistent testing patterns

## Migration Strategy

### Phase 1: Flagship Implementation ✅
- **distributions** module successfully reorganized as proof-of-concept
- Three files created with proper separation of concerns
- Original file backed up as `.backup`

### Phase 2: Systematic Rollout
Apply this pattern to remaining test modules:
- basic_stats
- cdf_functions  
- pmf_pdf_functions
- ppf_functions
- error_functions
- helper_functions
- cdf_pdf_integration
- sampling_gen

### Phase 3: Legacy Cleanup
- Remove original monolithic test files after validation
- Update test runner configurations
- Update documentation references

## Quality Assurance

### Pre-Migration Checklist
- [ ] Create module folder: `tests/core/[module_name]/`
- [ ] Extract SCIPY VALIDATION TESTS to `scipy_validation_test.gd`
- [ ] Extract MATHEMATICAL PROPERTY TESTS to `mathematical_property_test.gd`
- [ ] Extract PARAMETER VALIDATION TESTS to `parameter_validation_test.gd`
- [ ] Create backup of original file: `[original_file].backup`
- [ ] Verify all test functions are properly categorized
- [ ] Run tests to ensure no functionality is lost

### Post-Migration Validation
- [ ] All tests pass in new structure
- [ ] No duplicate test functions across files
- [ ] Proper class names and file headers
- [ ] Section headers are correct
- [ ] Test coverage is maintained

## Exception Handling

### Mixed Content
When original files contain mixed test types within sections:
1. **Analyze** each test function individually
2. **Categorize** based on actual test behavior, not section placement
3. **Relocate** to appropriate specialized file
4. **Document** any ambiguous cases for review

### Specialized Test Files
Integration or end-to-end test files may require custom handling:
1. **Preserve** specialized functionality
2. **Add** standard sections where appropriate
3. **Document** deviations from standard pattern

## Success Metrics

### Quantitative Measures
- **File Size Reduction**: Individual test files < 500 lines
- **Test Execution Speed**: Faster targeted test runs
- **Code Coverage**: Maintained at 100% for critical functions

### Qualitative Measures  
- **Developer Experience**: Easier navigation and maintenance
- **Code Review Efficiency**: Focused changes in specialized files
- **Testing Standards Compliance**: 100% adherence to three-section pattern

## Policy Enforcement

This policy is **MANDATORY** for all new test modules and **REQUIRED** for existing modules during the migration phase. Any deviations must be documented and approved through the standard review process.

**Effective Date**: Immediate
**Review Date**: After Phase 2 completion
**Policy Owner**: Development Team 