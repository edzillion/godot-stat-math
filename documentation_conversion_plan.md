# Documentation Conversion Plan for Godot Stat Math

## Overview
Convert all GDScript comments to proper docstrings using Sphinx-compatible formatting for Godot 4.1+ auto-documentation generation. The goal is to match the official Godot documentation style exactly.

## Conversion Status

### ✅ Completed Modules
- `basic_stats.gd` - Core statistical functions (mean, median, variance, etc.)
- `error_functions.gd` - Error function calculations (erf, erfc, gamma, etc.)
- `helper_functions.gd` - Foundation mathematical utilities (combinatorial, gamma, beta, preprocessing)
- `distributions.gd` - Random number generation for statistical distributions
- `cdf_functions.gd` - Cumulative distribution functions for various distributions
- `ppf_functions.gd` - Inverse CDF/quantile functions for various distributions
- `pmf_pdf_functions.gd` - Probability mass/density functions for discrete distributions
- `sampling_gen.gd` - Advanced sampling techniques and quasi-random sequences

### 🎉 Phase 1 Complete! 
**All 8 core modules have been converted to proper Sphinx documentation format**

## Proper Sphinx Documentation Format

Based on the [official Godot Animation class documentation](https://docs.godotengine.org/en/stable/classes/class_animation.html), we need to follow the Sphinx format exactly.

### Class-level Documentation
```gdscript
class_name ClassName extends RefCounted

## Class Description
##
## Detailed explanation of what this class provides.
## Mathematical context when relevant.
##
## Note: Important usage information
```

### Function Documentation Template (Sphinx Style)
```gdscript
## Brief function description.
##
## Detailed explanation including mathematical context.
## Additional details on separate lines.
## 
## Special Cases:
## case1 - description
## case2 - description
static func function_name(param: Type) -> ReturnType:
```

## Key Format Requirements

### ❌ AVOID (BBCode over-formatting):
- `[br]` for line breaks
- `[b]bold[/b]` formatting
- `[code]inline[/code]` formatting in descriptions
- `[codeblock]...[/codeblock]` example blocks
- Structured parameter/return sections

### ✅ USE (Clean Sphinx format):
- Simple paragraph breaks with blank lines
- Plain text descriptions
- Minimal formatting
- Mathematical symbols using Unicode
- Clear, concise descriptions matching official docs

## Priority Order

1. **helper_functions.gd** - Foundation mathematical functions
2. **distributions.gd** - Core random generation capabilities
3. **cdf_functions.gd** - Distribution functions
4. **ppf_functions.gd** - Inverse distribution functions
5. **pmf_pdf_functions.gd** - Probability functions
6. **sampling_gen.gd** - Advanced sampling (largest file)

## Organizational Groups

### helper_functions.gd
- Combinatorial Functions
- Gamma & Beta Functions  
- Special Mathematical Functions
- Array Utilities

### distributions.gd
- Integer Distributions (Bernoulli, Binomial, etc.)
- Continuous Distributions (Normal, Exponential, etc.)
- Custom/Game-specific Distributions

### cdf_functions.gd
- Discrete CDFs
- Continuous CDFs
- Helper/Utility CDFs

### Others
Similar logical groupings based on function purpose and mathematical relationships.

## Documentation Goals

1. **Sphinx Compatibility** - Generate docs identical to official Godot docs
2. **Mathematical Context** - Formulas and theory where relevant
3. **Practical Usage** - Game development focused descriptions
4. **Concise Clarity** - Clear, brief descriptions without over-formatting
5. **Consistency** - Uniform style across all modules

## Documentation Format Standards

### Formatting Guidelines
- ✅ Use clean, minimal Sphinx-compatible docstrings
- ✅ Mathematical notation formatted with `[code]` BBCode tags
- ✅ Unicode mathematical symbols (μ, σ, ∞, etc.)
- ✅ Simple paragraph breaks with blank lines
- ✅ Concise descriptions matching official Godot documentation style
- ✅ Logical organization with section headers

### Cross-Reference Format
- Function references: `[StatMath.HelperFunctions.sanitize_numeric_array]`
- Class references: `[ClassName]`
- Method references: `[ClassName.method_name]`

## Phase 1: Content Conversion ✅ COMPLETE

**Successfully converted all core modules:**
- Added proper class-level documentation explaining purpose and usage
- Converted function-level documentation with mathematical formulas
- Applied proper BBCode formatting for mathematical expressions
- Added logical organization and section headers
- Maintained consistent style across all modules

**Key Achievements:**
- 8 core modules totaling ~3,500 lines of code
- 100+ individual functions documented
- Comprehensive mathematical notation using `[code]` tags
- Clean, professional documentation matching Godot standards

## Phase 2: Cross-Reference Pass 🚧 NEXT

**Remaining Tasks:**
1. **Identify Cross-References**: Scan all converted modules for references to other functions/classes
2. **Add Linking Syntax**: Convert plain text references to proper linking format
3. **Validate Links**: Ensure all cross-references point to valid functions

**Examples of conversions needed:**
- `StatMath.HelperFunctions.sanitize_numeric_array()` → `[StatMath.HelperFunctions.sanitize_numeric_array]`
- `BasicStats.mean()` → `[BasicStats.mean]`
- `normal distribution CDF` → `[CdfFunctions.normal_cdf]`
- `binomial_pmf` references → `[PmfPdfFunctions.binomial_pmf]`

## GitHub Pages Integration Plan

### Documentation Generation Pipeline
1. **Sphinx Documentation Generator**
   - Configure Sphinx to parse GDScript docstrings
   - Apply official Godot documentation theme
   - Generate HTML documentation from docstrings

2. **GitHub Actions CI/CD**
   - Automated documentation generation on commits
   - Integration with test results and performance benchmarks
   - Deployment to GitHub Pages

3. **Unified Documentation Site**
   - API documentation (from docstrings)
   - Test coverage reports
   - Performance benchmark results
   - Usage examples and tutorials

### Target Structure
```
docs/
├── api/           # Generated from docstrings
├── tests/         # Test coverage reports  
├── performance/   # Benchmark results
└── examples/      # Usage guides
```

## Next Steps
1. ✅ ~~Complete conversion of all core modules~~ **DONE!**
2. 🚧 Complete cross-reference linking pass
3. 🚧 Set up Sphinx documentation generation system
4. 🚧 Configure GitHub Actions for automated deployment
5. 🚧 Create comprehensive API documentation site

## Summary
**Major milestone achieved!** All core statistical functions now have professional, 
Sphinx-compatible documentation that will generate beautiful API docs matching 
the official Godot documentation style. Ready to proceed with cross-reference 
linking and automated documentation generation. 