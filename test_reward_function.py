#!/usr/bin/env python3
"""
Test script for molecule generation reward function.
Tests format checking, validity, and property calculations.
"""

from verl.utils.reward_score.molecule_generation import compute_score

def test_case(name, response_str, expected_range=None):
    """Run a test case and print results."""
    print("=" * 80)
    print(f"Test Case: {name}")
    print("=" * 80)
    print(f"Input: {response_str[:200]}...")
    print()
    
    result = compute_score(response_str, "")
    
    print("Results:")
    print(f"  Total Score: {result['score']:.2f}")
    print(f"  Accuracy: {result['acc']}")
    print(f"  Predicted SMILES: {result.get('pred', 'N/A')}")
    
    if 'format' in result:
        print(f"\nComponent Scores:")
        print(f"  Format:   {result.get('format', 0):.2f}")
        print(f"  Validity: {result.get('validity', 0):.2f}")
        print(f"  QED:      {result.get('qed', 0):.2f}")
        print(f"  SA:       {result.get('sa', 0):.2f}")
        print(f"  LogP:     {result.get('logp', 0):.2f}")
        print(f"  MW:       {result.get('mw', 0):.2f}")
        print(f"  TPSA:     {result.get('tpsa', 0):.2f}")
        print(f"  Lipinski: {result.get('lipinski', 0):.2f}")
        print(f"  EGFR:     {result.get('egfr', 0):.2f}")
    
    if expected_range:
        if expected_range[0] <= result['score'] <= expected_range[1]:
            print(f"\n✅ PASS: Score {result['score']:.2f} is in expected range {expected_range}")
        else:
            print(f"\n❌ FAIL: Score {result['score']:.2f} is NOT in expected range {expected_range}")
    
    print()
    return result

def main():
    print("\n" + "=" * 80)
    print("MOLECULE GENERATION REWARD FUNCTION TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Perfect format error (no SMILES tags)
    test_case(
        "1. No SMILES Tags",
        "This is just text without SMILES tags.",
        expected_range=(-5.0, -5.0)
    )
    
    # Test 2: Multiple SMILES tags (format error)
    test_case(
        "2. Multiple SMILES Tags",
        "<SMILES> CCO </SMILES> and <SMILES> CCN </SMILES>",
        expected_range=(-5.0, -5.0)
    )
    
    # Test 3: Empty SMILES (format error)
    test_case(
        "3. Empty SMILES",
        "Here is a molecule: <SMILES>  </SMILES> .",
        expected_range=(-5.0, -5.0)
    )
    
    # Test 4: Invalid SMILES (format OK but invalid molecule)
    test_case(
        "4. Invalid SMILES String",
        "Here is a molecule: <SMILES> XYZABC123 </SMILES> .",
        expected_range=(-4.0, -2.0)  # format(+1) + validity(-3)
    )
    
    # Test 5: Valid but simple molecule (ethanol)
    test_case(
        "5. Simple Valid Molecule (Ethanol)",
        "Here is a potential molecule: <SMILES> CCO </SMILES> .",
        expected_range=(0.0, 5.0)  # Valid but probably poor drug properties
    )
    
    # Test 6: Aspirin (good drug-like properties)
    test_case(
        "6. Aspirin (Good Drug-like Molecule)",
        "Here is a potential molecule: <SMILES> CC(=O)Oc1ccccc1C(=O)O </SMILES> .",
        expected_range=(4.0, 8.0)  # Should have good drug-like properties
    )
    
    # Test 7: EGFR inhibitor-like molecule (from training data example)
    test_case(
        "7. EGFR Inhibitor-like Molecule",
        "Here is a potential molecule: <SMILES> C=CC(=O)Nc1cc(Nc2nccc(-c3cnc4c(C)cccn34)n2)c(OC)cc1N(C)CCN(C)C </SMILES> .",
        expected_range=(5.0, 10.0)  # Should have excellent properties
    )
    
    # Test 8: Format variations (case insensitive)
    test_case(
        "8. Lowercase SMILES Tags",
        "Here is a potential molecule: <smiles> CCO </smiles> .",
        expected_range=(0.0, 5.0)
    )
    
    # Test 9: Extra whitespace in SMILES
    test_case(
        "9. SMILES with Extra Whitespace",
        "<SMILES>   CC(=O)Oc1ccccc1C(=O)O   </SMILES>",
        expected_range=(4.0, 8.0)
    )
    
    # Test 10: Complex EGFR inhibitor
    test_case(
        "10. Complex EGFR Inhibitor",
        "Here is a potential molecule: <SMILES> COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1 </SMILES> .",
        expected_range=(5.0, 10.0)
    )
    
    # Summary
    print("=" * 80)
    print("TEST SUITE COMPLETED")
    print("=" * 80)
    print("\nReward Score Ranges:")
    print("  -5.0:        Format error (no/wrong SMILES tags)")
    print("  -4.0 to -2.0: Valid format but invalid SMILES")
    print("  0.0 to 3.0:   Valid molecule, poor drug properties")
    print("  3.0 to 5.0:   Valid molecule, moderate properties")
    print("  5.0 to 7.0:   Good drug-like molecule")
    print("  7.0 to 10.0:  Excellent drug-like molecule")
    print()

if __name__ == "__main__":
    main()

