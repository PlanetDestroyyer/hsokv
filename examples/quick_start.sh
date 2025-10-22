#!/bin/bash

echo "H-SOKV Quick Start Examples"
echo "=========================="
echo ""

echo "1. Test (2 minutes):"
echo "   python hsokv.py --preset quick_test --task classification --visualize"
echo ""

echo "2. Demo with auto-generated corpus (15 minutes):"
echo "   python hsokv.py --preset demo --task language_model --corpus-size medium --visualize"
echo ""

echo "3. Full research run (30+ minutes):"
echo "   python hsokv.py --preset research --task language_model --visualize"
echo ""

echo "4. Using your own corpus:"
echo "   python hsokv.py --task language_model --lm-corpus input.txt --visualize"
echo ""

echo "Run any of these commands above!"