#!/usr/bin/env python
"""
Cogniguard Runner Script
========================

This script provides a simplified way to run the Cogniguard application.
"""

import os
import sys
from cogniguard.app import main

if __name__ == "__main__":
    # Ensure models directory exists
    os.makedirs('cogniguard/models', exist_ok=True)
    
    # Run the main function
    main() 