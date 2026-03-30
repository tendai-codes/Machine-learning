# Market Basket Optimisation Using Apriori and Eclat

## Problem

This project analysed retail transaction data to identify frequently co-occurring product combinations that could support cross-selling and product placement strategies.

## Objective

- Identify frequent itemsets within transaction-level purchase data
- Generate association rules between commonly purchased products
- Apply Apriori and Eclat algorithms to discover purchasing patterns
- Compare rule structures produced by both association rule methods

## Approach

- Converted transaction records into itemset format suitable for association rule mining
- Applied the Apriori algorithm to extract frequent itemsets based on minimum support thresholds
- Implemented the Eclat algorithm to identify frequent product combinations using transaction intersections
- Examined generated association rules using support, confidence, and lift metrics

## Key Findings

- Both Apriori and Eclat identified recurring product combinations within transaction data
- Association rules revealed interpretable relationships between commonly co-purchased items
- Support, confidence, and lift metrics enabled structured comparison of rule strength across algorithms
- Overlapping frequent itemsets were observed between Apriori and Eclat outputs

## Notebooks
- `Market basket optimisation_Apriori.ipynb`
- `Market basket optimisation _Eclat.ipynb`
- `Market_Basket_Optimisation.csv`

