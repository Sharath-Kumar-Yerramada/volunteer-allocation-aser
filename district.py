import pandas as pd
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

CSV_DIRECTORY = r'C:\Users\ADMIN\Downloads\states csv aser' 
# Make sure these column names EXACTLY match your CSV files
DISTRICT_COLUMN = 'District' 
READING_PROFICIENCY_COLUMN = '% Children who can read Std II level text (Std III to V: Learning levels)'
MATH_PROFICIENCY_COLUMN = '% Children who can do at least subtraction (Std III to V: Learning levels)'

# --- Optimization Strategy ---
# Set USE_COMPOSITE_SCORE to True to use both Reading and Math need, False to use Reading only
# Using True is generally recommended for a more holistic view if math data is reliable.
USE_COMPOSITE_SCORE = True 

# Weights for Reading and Math need (used only if USE_COMPOSITE_SCORE is True)
# Ensure these sum to 1.0. Example: 0.6 for Reading, 0.4 for Math if reading is prioritized.
WEIGHT_READING = 0.5      
WEIGHT_MATH = 0.5         

# --- Resource Allocation Parameters ---
# Average hours per district used to set the total budget FOR EACH STATE
# Increasing this gives states more hours, potentially reducing the "bang-bang" effect.
AVG_HOURS_PER_DISTRICT = 2.0 

# Minimum hours ANY district within a state can receive. 
# Increasing this ensures a higher baseline for all, but reduces hours for the neediest.
MIN_HOURS_PER_DISTRICT = 1.0

# Maximum hours ANY district within a state can receive.
# Decreasing this spreads hours more but reduces focus on the neediest.
MAX_HOURS_PER_DISTRICT = 5.0

# --- Technical Parameters ---
SAFE_GAIN_FLOOR = 0.001 # Avoid division by zero for low/zero proficiency

# --- Visualization ---
NUM_DISTRICTS_TO_SHOW = 15           # For top/bottom lists and plots
PLOT_STYLE = 'seaborn-v0_8-whitegrid' # Plotting style

# --- Weight Normalization (Safety Check) ---
if USE_COMPOSITE_SCORE:
    if not np.isclose(WEIGHT_READING + WEIGHT_MATH, 1.0):
        print("Warning: Composite score weights do not sum to 1.0. Normalizing...")
        total_weight = WEIGHT_READING + WEIGHT_MATH
        if total_weight > 0:
            WEIGHT_READING /= total_weight
            WEIGHT_MATH /= total_weight
        else: # Avoid division by zero if both weights are zero
            print("Error: Both composite weights are zero. Setting to 0.5 each.")
            WEIGHT_READING = 0.5
            WEIGHT_MATH = 0.5
        print(f"Using normalized weights: Reading={WEIGHT_READING:.2f}, Math={WEIGHT_MATH:.2f}")
    else:
         print(f"Using composite score weights: Reading={WEIGHT_READING:.2f}, Math={WEIGHT_MATH:.2f}")
else:
    print("Using Reading proficiency ONLY for need score.")


# --- Functions ---

def load_prepare_and_combine_data(directory, district_col, read_prof_col, math_prof_col):
    """
    Loads all CSVs from directory, cleans data (removes invalid rows), 
    calculates inverse gains, composite score (optional), and combines.
    """
    all_district_data = []
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    if not csv_files:
        print(f"Error: No CSV files found in directory '{directory}'.")
        return None

    print(f"\nFound {len(csv_files)} CSV files. Processing...")

    # Define required columns based on whether composite score is used
    required_cols = [district_col, read_prof_col]
    if USE_COMPOSITE_SCORE:
        required_cols.append(math_prof_col)

    processed_files_count = 0
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        # Extract state ID robustly (handle potential dots in filename before extension)
        state_id = os.path.splitext(filename)[0] 
        
        try:
            df = pd.read_csv(filepath)
            print(f"  Processing '{filename}' (State ID: {state_id})...")

            # --- Basic Validation ---
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"    Warning: Skipping '{filename}'. Missing required columns: {missing_cols}")
                continue
                
            if df.empty:
                 print(f"    Warning: Skipping '{filename}'. File is empty.")
                 continue

            # --- Filter out State Summary Row (before numeric conversion) ---
            df_filtered = df.copy() 
            identified_summary = False
            if len(df_filtered) > 1: 
                potential_summary_name = df_filtered.iloc[-1][district_col]
                # Basic check: Does the last row district name look like a state name? (heuristic)
                # You might need a more robust check if district names can be identical to state names
                if isinstance(potential_summary_name, str) and potential_summary_name.lower() in state_id.lower() or state_id.lower() in potential_summary_name.lower():
                    # Filter out rows where District name matches the last row's district name
                    df_filtered = df_filtered[df_filtered[district_col] != potential_summary_name]
                    identified_summary = True
                
                if len(df_filtered) == len(df) and identified_summary: # Filtering didn't change length but we thought we found one
                     print(f"    Warning: Potential summary row '{potential_summary_name}' found but filtering didn't remove it. Check for duplicate district names.")
                elif not identified_summary:
                     print(f"    Warning: Could not confidently identify summary row in '{filename}'. Keeping all rows. Check last row's '{district_col}'.")
                # If filtering worked, df_filtered is now shorter.

            if df_filtered.empty: 
                print(f"    Warning: Skipping '{filename}'. No district data after attempting summary row removal.")
                continue

            # --- Convert Proficiency Columns to Numeric ---
            df_filtered[read_prof_col] = pd.to_numeric(df_filtered[read_prof_col], errors='coerce')
            if USE_COMPOSITE_SCORE:
                df_filtered[math_prof_col] = pd.to_numeric(df_filtered[math_prof_col], errors='coerce')

            # --- CRITICAL FIX: Filter out invalid proficiency data ---
            initial_district_count = len(df_filtered)
            if USE_COMPOSITE_SCORE:
                # Keep rows where BOTH reading and math are valid (> 0 and not NaN)
                df_valid = df_filtered[(df_filtered[read_prof_col].notna()) & (df_filtered[read_prof_col] > 0) &
                                       (df_filtered[math_prof_col].notna()) & (df_filtered[math_prof_col] > 0)].copy()
            else:
                # Keep rows where reading is valid (> 0 and not NaN)
                df_valid = df_filtered[(df_filtered[read_prof_col].notna()) & (df_filtered[read_prof_col] > 0)].copy()
            
            removed_count = initial_district_count - len(df_valid)
            if removed_count > 0:
                print(f"    Removed {removed_count} districts from '{state_id}' due to missing/invalid (<0) proficiency.")

            if df_valid.empty:
                print(f"    No valid district data remaining for '{state_id}' after cleaning proficiency.")
                continue

            # --- Calculate Inverse Gains & Composite Score (on df_valid) ---
            df_valid['State'] = state_id

            # Reading Inverse Gain
            g_read_fraction = df_valid[read_prof_col] / 100.0
            g_read_safe = g_read_fraction.clip(lower=SAFE_GAIN_FLOOR, upper=1.0) # More concise way
            df_valid['g_read_inv'] = 1.0 / g_read_safe

            # Need Score Calculation
            if USE_COMPOSITE_SCORE:
                 # Math Inverse Gain
                 g_math_fraction = df_valid[math_prof_col] / 100.0
                 g_math_safe = g_math_fraction.clip(lower=SAFE_GAIN_FLOOR, upper=1.0)
                 df_valid['g_math_inv'] = 1.0 / g_math_safe
                 
                 # Composite Need Score
                 df_valid['need_score'] = (WEIGHT_READING * df_valid['g_read_inv']) + (WEIGHT_MATH * df_valid['g_math_inv'])
            else:
                 # If not using composite, the 'need score' is just the reading inverse gain
                 df_valid['need_score'] = df_valid['g_read_inv']
                 df_valid['g_math_inv'] = np.nan # Add NaN column for consistency if needed later

            all_district_data.append(df_valid)
            processed_files_count += 1

        except FileNotFoundError:
            print(f"    Error: File not found at {filepath}")
        except Exception as e:
            print(f"    An error occurred processing '{filename}': {e}")

    if not all_district_data:
        print("Error: No valid district data could be loaded and processed from any CSV file.")
        return None

    # Combine all dataframes
    combined_df = pd.concat(all_district_data, ignore_index=True)
    print(f"\nSuccessfully combined and cleaned data from {processed_files_count} files.")
    print(f"Total number of districts included in analysis: {len(combined_df)}")

    # Final check for calculation issues
    if 'need_score' not in combined_df.columns or combined_df['need_score'].isnull().any() or np.isinf(combined_df['need_score']).any():
         print("Warning: Null or Infinite values found in final 'need_score'. Check calculations and SAFE_GAIN_FLOOR.")
         
    return combined_df


def plot_descriptive_stats(df, proficiency_col, district_col):
    """Generates plots for descriptive statistics (using combined CLEANED data)."""
    plt.style.use(PLOT_STYLE)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df[proficiency_col].dropna(), kde=True, bins=30) 
    plt.title(f'Distribution of District-Level Reading Proficiency (Valid Data Only)')
    plt.xlabel(f'{proficiency_col} (%)')
    plt.ylabel('Number of Districts')
    plt.tight_layout()
    plt.savefig('dist_reading_proficiency_histogram_cleaned.png')
    print("Saved: dist_reading_proficiency_histogram_cleaned.png")
    plt.show()

    lowest_g_i = df.nsmallest(NUM_DISTRICTS_TO_SHOW, proficiency_col)
    if not lowest_g_i.empty:
        lowest_g_i['Display_Label'] = lowest_g_i[district_col] + ' (' + lowest_g_i['State'] + ')'
        plt.figure(figsize=(10, 7))
        sns.barplot(x=proficiency_col, y='Display_Label', data=lowest_g_i, palette='viridis')
        plt.title(f'Top {NUM_DISTRICTS_TO_SHOW} Districts with Lowest Reading Proficiency (Valid Data)')
        plt.xlabel(f'{proficiency_col} (%)')
        plt.ylabel('District (State)')
        plt.tight_layout()
        plt.savefig('dist_lowest_reading_proficiency_barchart_cleaned.png')
        print("Saved: dist_lowest_reading_proficiency_barchart_cleaned.png")
        plt.show()
    else:
        print("Skipping lowest proficiency plot - no valid data.")


def plot_optimization_results(df, optimized_hours_col, proficiency_col, district_col):
    """Generates plots for STATE-WISE optimization results."""
    plt.style.use(PLOT_STYLE)

    if optimized_hours_col not in df.columns or df[optimized_hours_col].isnull().all():
        print(f"Warning: Cannot generate optimization plots. Column '{optimized_hours_col}' missing or empty.")
        return

    plt.figure(figsize=(10, 6))
    sns.histplot(df[optimized_hours_col].dropna(), kde=True, bins=20)
    plt.title(f'Optimized Hours Distribution (Min={MIN_HOURS_PER_DISTRICT}, Max={MAX_HOURS_PER_DISTRICT}, Avg={AVG_HOURS_PER_DISTRICT})')
    plt.xlabel('Optimized Volunteer Hours (x_i*)')
    plt.ylabel('Number of Districts')
    plt.tight_layout()
    plt.savefig('dist_optimized_hours_histogram_statewise.png')
    print("Saved: dist_optimized_hours_histogram_statewise.png")
    plt.show()

    highest_x_i = df.dropna(subset=[optimized_hours_col]).nlargest(NUM_DISTRICTS_TO_SHOW, optimized_hours_col)
    if not highest_x_i.empty:
        highest_x_i['Display_Label'] = highest_x_i[district_col] + ' (' + highest_x_i['State'] + ')'
        plt.figure(figsize=(10, 7))
        sns.barplot(x=optimized_hours_col, y='Display_Label', data=highest_x_i, palette='magma')
        plt.title(f'Top {NUM_DISTRICTS_TO_SHOW} Districts Receiving Most Volunteer Hours (State-wise Opt.)')
        plt.xlabel('Optimized Volunteer Hours (x_i*)')
        plt.ylabel('District (State)')
        plt.tight_layout()
        plt.savefig('dist_highest_allocated_hours_barchart_statewise.png')
        print("Saved: dist_highest_allocated_hours_barchart_statewise.png")
        plt.show()
    else:
         print("Skipping highest allocated hours plot - no valid optimized hours found.")

    plt.figure(figsize=(12, 7)) 
    plot_data = df.dropna(subset=[optimized_hours_col, proficiency_col, 'State'])
    if not plot_data.empty:
        sns.scatterplot(x=proficiency_col, y=optimized_hours_col, data=plot_data, alpha=0.7, hue='State', legend='full', s=50) # Slightly larger points
        plt.title('Optimized Hours (State-wise) vs. Reading Proficiency')
        plt.xlabel(f'{proficiency_col} (%)')
        plt.ylabel('Optimized Volunteer Hours (x_i*)')
        plt.axhline(MIN_HOURS_PER_DISTRICT, color='red', linestyle='--', label=f'Min Hours ({MIN_HOURS_PER_DISTRICT})')
        plt.axhline(MAX_HOURS_PER_DISTRICT, color='green', linestyle='--', label=f'Max Hours ({MAX_HOURS_PER_DISTRICT})')
        # Adjust legend position and plot margins
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title='State') 
        plt.subplots_adjust(right=0.80) # Make space for the legend
        # plt.tight_layout(rect=[0, 0, 0.85, 1]) # Alternative way to adjust
        plt.savefig('dist_hours_vs_proficiency_scatter_statewise.png')
        print("Saved: dist_hours_vs_proficiency_scatter_statewise.png")
        plt.show()
    else:
        print("Skipping scatter plot - no valid data points found after dropping NaNs.")


# --- Main Execution ---
if __name__ == "__main__":
    
    print("Starting Volunteer Allocation Optimization Script...")
    print("-" * 50)
    print("Configuration:")
    print(f"  CSV Directory: {os.path.abspath(CSV_DIRECTORY)}")
    print(f"  District Column: '{DISTRICT_COLUMN}'")
    print(f"  Reading Proficiency Column: '{READING_PROFICIENCY_COLUMN}'")
    if USE_COMPOSITE_SCORE:
        print(f"  Math Proficiency Column: '{MATH_PROFICIENCY_COLUMN}'")
        print(f"  Using Composite Need Score: Yes (Read Weight={WEIGHT_READING:.2f}, Math Weight={WEIGHT_MATH:.2f})")
    else:
        print("  Using Composite Need Score: No (Reading Only)")
    print(f"  Avg Hours per District (for State Budget): {AVG_HOURS_PER_DISTRICT}")
    print(f"  Min Hours Constraint per District: {MIN_HOURS_PER_DISTRICT}")
    print(f"  Max Hours Constraint per District: {MAX_HOURS_PER_DISTRICT}")
    print("-" * 50)

    # 1. Load, Prepare, Clean, and Combine Data 
    df_districts = load_prepare_and_combine_data(
        CSV_DIRECTORY, 
        DISTRICT_COLUMN, 
        READING_PROFICIENCY_COLUMN, 
        MATH_PROFICIENCY_COLUMN 
    )

    if df_districts is not None and not df_districts.empty:
        
        # 2. Descriptive Statistics and Plots (on combined CLEANED initial data)
        print("\n--- Descriptive Statistics (All Valid Districts Pre-Optimization) ---")
        if READING_PROFICIENCY_COLUMN in df_districts.columns:
             print(f"Reading Proficiency ({READING_PROFICIENCY_COLUMN}):")
             print(df_districts[READING_PROFICIENCY_COLUMN].describe().to_string()) # Use to_string for better console format
             if USE_COMPOSITE_SCORE and MATH_PROFICIENCY_COLUMN in df_districts.columns and df_districts[MATH_PROFICIENCY_COLUMN].notna().any():
                 print(f"\nMath Proficiency ({MATH_PROFICIENCY_COLUMN}):")
                 print(df_districts[MATH_PROFICIENCY_COLUMN].describe().to_string())
             if 'need_score' in df_districts.columns and df_districts['need_score'].notna().any():
                 print(f"\nCalculated Need Score {'(Composite)' if USE_COMPOSITE_SCORE else '(Reading Only)'}:")
                 print(df_districts['need_score'].describe().to_string())
             
             # Generate descriptive plots
             plot_descriptive_stats(df_districts, READING_PROFICIENCY_COLUMN, DISTRICT_COLUMN)
        else:
             print(f"Error: Column '{READING_PROFICIENCY_COLUMN}' not found for descriptive stats.")
             exit()

        # 3. State-by-State Optimization
        print("\n--- Starting State-by-State Optimization ---")
        # Ensure 'State' column exists
        if 'State' not in df_districts.columns:
            print("Error: 'State' column not found in the combined data. Cannot proceed with state-wise optimization.")
            exit()
            
        unique_states = sorted(df_districts['State'].unique()) 
        print(f"Found {len(unique_states)} states with valid data for optimization: {unique_states}")

        df_districts['Optimized_Hours_Statewise'] = np.nan
        df_districts['Priority_Score_Statewise'] = np.nan # Based on objective function coeff * hours
        
        optimization_success_count = 0
        optimization_fail_count = 0
        skipped_feasibility_count = 0

        for state in unique_states:
            print(f"\n--- Optimizing for State: {state} ---")
            # Filter data for the current state
            df_state = df_districts[df_districts['State'] == state].copy()

            if df_state.empty: 
                print("  Skipping: No districts found for this state after initial filtering.")
                continue

            # Check for valid need score data within the state
            if 'need_score' not in df_state.columns or df_state['need_score'].isnull().any() or np.isinf(df_state['need_score']).any():
                print("  Warning: Skipping optimization for this state due to missing, null, or infinite 'need_score' values.")
                optimization_fail_count +=1
                continue

            N_state = len(df_state)
            H_total_state = AVG_HOURS_PER_DISTRICT * N_state
            need_score_values_state = df_state['need_score'].values

            print(f"  Number of districts (N_state): {N_state}")
            print(f"  Total available hours (H_total_state): {H_total_state:.2f}")

            # --- LP Setup for the state ---
            # Objective: Maximize sum(x_i * need_score) -> Minimize sum(x_i * -need_score)
            c_state = -need_score_values_state 
            # Constraint 1: sum(x_i) <= H_total_state
            A_ub_state = [np.ones(N_state)]
            b_ub_state = [H_total_state]
            # Constraint 2: Bounds Min <= x_i <= Max
            bounds_state = [(MIN_HOURS_PER_DISTRICT, MAX_HOURS_PER_DISTRICT)] * N_state

            # --- Feasibility Check (Diagnostic) ---
            min_hours_needed_state = N_state * MIN_HOURS_PER_DISTRICT
            if H_total_state < min_hours_needed_state - 1e-9: # Allow for small floating point tolerance
                print(f"  ERROR: INFEASIBLE! Total available hours ({H_total_state:.2f}) is less than minimum required ({min_hours_needed_state:.2f}).")
                print(f"  Skipping optimization for state {state}.")
                optimization_fail_count += 1
                skipped_feasibility_count += 1
                continue # Skip to the next state

            # --- Solve ---
            try:
                # Use 'highs' for better performance and reliability with bounds
                result_state = linprog(c_state, A_ub=A_ub_state, b_ub=b_ub_state, bounds=bounds_state, method='highs')

                if result_state.success:
                    print("  Optimization successful!")
                    optimization_success_count += 1
                    optimal_hours_state = result_state.x
                    
                    # Store results back into the main DataFrame using the state dataframe's index
                    df_districts.loc[df_state.index, 'Optimized_Hours_Statewise'] = np.round(optimal_hours_state, 2)
                    df_districts.loc[df_state.index, 'Priority_Score_Statewise'] = np.round(optimal_hours_state * need_score_values_state, 2)
                    
                    state_allocated = df_districts.loc[df_state.index, 'Optimized_Hours_Statewise'].sum()
                    print(f"  Total hours allocated in {state}: {state_allocated:.2f} (Available: {H_total_state:.2f})")
                    # Check if budget was binding
                    if np.isclose(state_allocated, H_total_state):
                        print("  State budget constraint was binding.")
                    # Check bounds saturation
                    at_min = np.sum(np.isclose(optimal_hours_state, MIN_HOURS_PER_DISTRICT))
                    at_max = np.sum(np.isclose(optimal_hours_state, MAX_HOURS_PER_DISTRICT))
                    print(f"  Districts at Min ({MIN_HOURS_PER_DISTRICT}): {at_min} / {N_state}")
                    print(f"  Districts at Max ({MAX_HOURS_PER_DISTRICT}): {at_max} / {N_state}")


                else:
                    print(f"  Optimization failed for state {state}.")
                    print(f"  Status: {result_state.status} ({result_state.message})")
                    optimization_fail_count += 1
            
            except ValueError as ve:
                 print(f"  ValueError during optimization setup/run for state {state}: {ve}")
                 print("  Check input arrays for NaNs/Infs not caught earlier. Skipping state.")
                 optimization_fail_count += 1
            except Exception as e:
                 print(f"  An unexpected error occurred during optimization for state {state}: {e}")
                 optimization_fail_count += 1


        # 4. Post-Optimization Analysis (using combined state-wise results)
        print("\n--- Overall Summary (Based on State-wise Optimizations) ---")
        print(f"States attempted: {len(unique_states)}")
        print(f"Successful state optimizations: {optimization_success_count}")
        print(f"Failed/Skipped state optimizations: {optimization_fail_count} ({skipped_feasibility_count} due to budget<min_req)")

        if optimization_success_count > 0:
            # Analyze results only from successful optimizations
            valid_results_df = df_districts.dropna(subset=['Optimized_Hours_Statewise'])
            
            total_allocated_overall = valid_results_df['Optimized_Hours_Statewise'].sum()
            print(f"\nTotal hours allocated across all successful states: {total_allocated_overall:.2f}")

            print("\nSummary statistics for Optimized Hours (State-wise results):")
            print(valid_results_df['Optimized_Hours_Statewise'].describe().to_string())

            valid_hours = valid_results_df['Optimized_Hours_Statewise']
            total_at_min = sum(np.isclose(valid_hours, MIN_HOURS_PER_DISTRICT))
            total_at_max = sum(np.isclose(valid_hours, MAX_HOURS_PER_DISTRICT))
            print(f"\nTotal districts at MIN bound ({MIN_HOURS_PER_DISTRICT} hours): {total_at_min} / {len(valid_hours)}")
            print(f"Total districts at MAX bound ({MAX_HOURS_PER_DISTRICT} hours): {total_at_max} / {len(valid_hours)}")

            # Define columns to display, adapt based on composite score usage
            display_cols = ['State', DISTRICT_COLUMN, READING_PROFICIENCY_COLUMN]
            if USE_COMPOSITE_SCORE and MATH_PROFICIENCY_COLUMN in valid_results_df.columns:
                 display_cols.append(MATH_PROFICIENCY_COLUMN)
            if 'need_score' in valid_results_df.columns:
                 display_cols.append('need_score') 
            display_cols.extend(['Optimized_Hours_Statewise', 'Priority_Score_Statewise'])
            
            print(f"\n--- Top {NUM_DISTRICTS_TO_SHOW} Districts by Allocated Hours (State-wise Opt.) ---")
            print(valid_results_df.nlargest(NUM_DISTRICTS_TO_SHOW, 'Optimized_Hours_Statewise')[display_cols].to_string())

            print(f"\n--- Bottom {NUM_DISTRICTS_TO_SHOW} Districts by Allocated Hours (State-wise Opt.) ---")
            # Show districts exactly at min or slightly above
            print(valid_results_df[np.isclose(valid_results_df['Optimized_Hours_Statewise'], MIN_HOURS_PER_DISTRICT)].nsmallest(NUM_DISTRICTS_TO_SHOW, 'Optimized_Hours_Statewise', keep='all')[display_cols].to_string())


            # 5. Plot Combined State-wise Optimization Results
            print("\nGenerating optimization result plots...")
            plot_optimization_results(df_districts, 'Optimized_Hours_Statewise', READING_PROFICIENCY_COLUMN, DISTRICT_COLUMN)

            # 6. Save Results
            output_filename = "districts_statewise_optimized_allocation_final.csv"
            print(f"\nSaving detailed results to '{output_filename}'...")
            try:
                # Select and order columns for saving
                save_cols = [DISTRICT_COLUMN, 'State', READING_PROFICIENCY_COLUMN]
                if USE_COMPOSITE_SCORE and MATH_PROFICIENCY_COLUMN in df_districts.columns:
                     save_cols.append(MATH_PROFICIENCY_COLUMN)
                # Include calculated need metrics for reference
                save_cols.extend(['g_read_inv'])
                if USE_COMPOSITE_SCORE and 'g_math_inv' in df_districts.columns:
                     save_cols.append('g_math_inv')
                if 'need_score' in df_districts.columns:
                     save_cols.append('need_score')
                # Add optimization results
                save_cols.extend(['Optimized_Hours_Statewise', 'Priority_Score_Statewise'])
                
                # Include other original columns if they exist and are desired (exclude intermediate calc columns)
                original_cols_to_keep = [col for col in df_districts.columns if col not in save_cols and col not in ['g_read_safe','g_math_safe']] 
                
                df_to_save = df_districts[original_cols_to_keep + save_cols].copy()
                df_to_save.sort_values(by=['State', DISTRICT_COLUMN], inplace=True) 

                df_to_save.to_csv(output_filename, index=False, float_format='%.2f') # Format floats
                print(f"Results successfully saved.")
            except Exception as e:
                print(f"\nError saving results to CSV: {e}")
        else:
            print("\nNo state optimizations were successful. Cannot generate summary plots or save results.")

    else:
        print("\nExiting script due to data loading/preparation errors.")
        
    print("\nScript finished.")
