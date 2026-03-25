"""
COMP 262 - Assignment 3, Exercise 1: Association Rules Cuisine Recommender
Student: Izzet Abidi (300898230)

Builds a non-personalized recommender that suggests top ingredient groups
for a given cuisine type using the Apriori algorithm on recipe data.

Pipeline:
1. Loads recipes.json (39,774 recipes across 20 cuisine types)
2. Performs basic data exploration (total recipes, cuisine counts)
3. Accepts a cuisine type from the user
4. Filters recipes for that cuisine, extracts ingredient lists
5. Runs Apriori with support = 100 / num_recipes, confidence = 0.5
6. Displays the most frequent ingredient group and all rules with lift > 2
7. Loops until the user types "exit"
"""

import json
import os
import sys
import pandas as pd
from apyori import apriori

# ----- Configuration -----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECIPES_FILE = os.path.join(SCRIPT_DIR, "recipes.json")


def load_recipes(filepath):
    """
    Loads the recipes JSON file into a pandas DataFrame.
    Each record has: id, cuisine, ingredients (list of strings).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    return df


def explore_data(df):
    """
    Performs and prints basic data exploration:
    - Total number of recipe instances
    - Number of unique cuisines
    - Table of cuisine types with recipe counts
    """
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)

    print(f"\nTotal number of recipe instances: {len(df)}")
    print(f"Number of unique cuisines: {df['cuisine'].nunique()}")

    # Cuisine breakdown table
    cuisine_counts = df["cuisine"].value_counts().reset_index()
    cuisine_counts.columns = ["Cuisine Type", "Number of Recipes"]
    print(f"\n{'Cuisine Type':<20} {'Number of Recipes':>18}")
    print("-" * 40)
    for _, row in cuisine_counts.iterrows():
        print(f"{row['Cuisine Type']:<20} {row['Number of Recipes']:>18}")

    print("-" * 40)
    print(f"{'TOTAL':<20} {len(df):>18}")

    return cuisine_counts


def run_apriori_for_cuisine(df, cuisine_type):
    """
    Filters the DataFrame for the given cuisine type, extracts ingredient
    lists as transactions, and runs the Apriori algorithm.

    Parameters:
    - support: 100 / total_recipes_for_cuisine (as specified in assignment)
    - confidence: 0.5
    - min_length: 2 (to get meaningful associations)

    Returns the list of RelationRecord objects from apyori.
    """
    # Filter for the selected cuisine
    cuisine_df = df[df["cuisine"].str.lower() == cuisine_type.lower()]
    num_recipes = len(cuisine_df)

    if num_recipes == 0:
        return None, 0

    # Extract ingredient lists as transactions
    transactions = cuisine_df["ingredients"].tolist()

    # Calculate support: 100 / total number of recipes for this cuisine
    min_support = 100 / num_recipes
    print(f"\nApriori parameters:")
    print(f"  Recipes for '{cuisine_type}': {num_recipes}")
    print(f"  Min support: 100 / {num_recipes} = {min_support:.4f}")
    print(f"  Min confidence: 0.5")

    # Run apriori
    results = list(apriori(
        transactions,
        min_support=min_support,
        min_confidence=0.5,
        min_length=2
    ))

    return results, num_recipes


def display_results(results, cuisine_type):
    """
    Displays:
    1. The top (most frequent) ingredient group — the first RelationRecord,
       which has the highest support.
    2. All association rules where lift > 2.
    """
    if not results:
        print(f"\nNo association rules found for '{cuisine_type}'.")
        return

    # ----- Top ingredient group (first record = highest support) -----
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: {cuisine_type.upper()}")
    print(f"{'='*60}")

    top_record = results[0]
    top_items = list(top_record.items)
    print(f"\n--- Most Frequent Ingredient Group ---")
    print(f"Ingredients: {', '.join(sorted(top_items))}")
    print(f"Support: {top_record.support:.4f}")

    # ----- All rules with lift > 2 -----
    print(f"\n--- Association Rules with Lift > 2 ---")
    print(f"{'Items (LHS)':<35} {'→':>3} {'Items (RHS)':<35} {'Conf':>6} {'Lift':>6}")
    print("-" * 90)

    rule_count = 0
    for record in results:
        for stat in record.ordered_statistics:
            if stat.lift > 2:
                lhs = ", ".join(sorted(stat.items_base)) if stat.items_base else "(all)"
                rhs = ", ".join(sorted(stat.items_add))
                print(f"{lhs:<35} {'→':>3} {rhs:<35} {stat.confidence:>6.3f} {stat.lift:>6.3f}")
                rule_count += 1

    if rule_count == 0:
        print("No rules found with lift > 2.")
    else:
        print(f"\nTotal rules with lift > 2: {rule_count}")


def main():
    print("=" * 60)
    print("Assignment 3, Exercise 1: Cuisine Ingredient Recommender")
    print("=" * 60)

    # Load and explore
    print(f"\nLoading recipes from: {RECIPES_FILE}")
    df = load_recipes(RECIPES_FILE)
    cuisine_counts = explore_data(df)

    # Get list of valid cuisines for validation
    valid_cuisines = set(df["cuisine"].str.lower().unique())

    # Interactive loop
    print("\n" + "=" * 60)
    print("Enter a cuisine type to get ingredient recommendations.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        user_input = input("\nEnter cuisine type: ").strip()

        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        if not user_input:
            print("Please enter a cuisine type.")
            continue

        if user_input.lower() not in valid_cuisines:
            print(f"We don't have recommendations for {user_input}")
            continue

        # Run apriori
        results, num_recipes = run_apriori_for_cuisine(df, user_input)

        if results is None:
            print(f"We don't have recommendations for {user_input}")
            continue

        # Display results
        display_results(results, user_input)


if __name__ == "__main__":
    main()
