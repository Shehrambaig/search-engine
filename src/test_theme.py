"""
Test script to preview the black & grey theme
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import the theme setup
from evaluation import setup_dark_theme

# Create output directory
Path('results/plots').mkdir(parents=True, exist_ok=True)

# Setup theme
colors = setup_dark_theme()

# Test Plot 1: Bar chart
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Config A', 'Config B', 'Config C', 'Config D']
values = [0.75, 0.82, 0.68, 0.90]

ax.bar(categories, values, color=colors['bars'][0], edgecolor='#404040', alpha=0.9)
ax.set_xlabel('Configuration', fontweight='bold')
ax.set_ylabel('Performance Score', fontweight='bold')
ax.set_title('Sample Evaluation Results', fontweight='bold', pad=20)
ax.grid(axis='y', alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('results/plots/theme_test.png', dpi=300, facecolor=colors['bg'])
plt.close()

print("✓ Theme test plot saved to: results/plots/theme_test.png")
print("\nTheme colors:")
print(f"  Background: {colors['bg']}")
print(f"  Foreground: {colors['fg']}")
print(f"  Bar colors: {colors['bars']}")
print("\nThe plots will use this black & grey theme when you run evaluation!")
