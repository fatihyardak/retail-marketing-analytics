import pandas as pd
import os
import numpy as np

def normalize_cols(df):
    """Normalize column names to lowercase with underscores"""
    df.columns = (df.columns
        .str.strip().str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)   
        .str.replace(r"_+", "_", regex=True)           
        .str.strip("_")                                
    )
    return df

def process_and_save_datasets():
    """Process raw datasets and save to processed folder"""
    
    # Get the project root directory (one level up from src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths relative to project root
    raw_dir = os.path.join(project_root, "datasets", "raw")
    processed_dir = os.path.join(project_root, "datasets", "processed")
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Reading from: {raw_dir}")
    print(f"Saving to: {processed_dir}")
    
    # Load raw datasets
    camp_desc = pd.read_csv(os.path.join(raw_dir, "campaign_desc.csv"))
    camp_table = pd.read_csv(os.path.join(raw_dir, "campaign_table.csv"))
    causal_data = pd.read_csv(os.path.join(raw_dir, "causal_data.csv"))
    coupon_redempt = pd.read_csv(os.path.join(raw_dir, "coupon_redempt.csv"))
    coupon = pd.read_csv(os.path.join(raw_dir, "coupon.csv"))
    hh_demographic = pd.read_csv(os.path.join(raw_dir, "hh_demographic.csv"))
    product = pd.read_csv(os.path.join(raw_dir, "product.csv"))
    transaction = pd.read_csv(os.path.join(raw_dir, "transaction_data.csv"))
    
    data_list = [camp_desc, camp_table, causal_data, coupon_redempt, 
                 coupon, hh_demographic, product, transaction]
    
    print("Normalizing column names...")
    
    # Normalizing column names
    for i, data in enumerate(data_list):
        data_list[i] = normalize_cols(data)
    
    # Unpack the processed dataframes
    camp_desc, camp_table, causal_data, coupon_redempt, \
    coupon, hh_demographic, product, transaction = data_list

    print("Renaming columns...")
    
    # Renaming columns
    hh_demographic.rename(columns={"household_key": "household_id"}, inplace=True)   
    camp_table.rename(columns={"household_key": "household_id"}, inplace=True)
    camp_desc.rename(columns={"campaign": "campaign_id"}, inplace=True)
    camp_table.rename(columns={"campaign": "campaign_id"}, inplace=True)
    coupon_redempt.rename(columns={"campaign": "campaign_id"}, inplace=True)
    coupon.rename(columns={"campaign": "campaign_id"}, inplace=True)
    transaction.rename(columns={"household_key": "household_id"}, inplace=True)
    coupon_redempt.rename(columns={"household_key": "household_id"}, inplace=True)
    
    print("Creating derived datasets...")
    
    # basket_fact dataset
    basket_fact = (transaction
                   .groupby(["household_id","basket_id","day"], as_index=False)
                   .agg(
                       n_items=("product_id","count"),
                       qty=("quantity","sum"),
                       sales=("sales_value","sum"),
                       retail_disc=("retail_disc","sum"),
                       coupon_disc=("coupon_disc","sum"),
                       coupon_match_disc=("coupon_match_disc","sum")
                   ))
    
    # basket_sum dataset
    t = transaction[["household_id", "basket_id", "day", "sales_value"]].copy()
    basket_sum = t.groupby("household_id").agg(
        total_sales=("sales_value", "sum"),
        basket_count=("basket_id", "nunique")
    ).reset_index()
    
    basket_sum["avg_basket_size"] = basket_sum["total_sales"] / basket_sum["basket_count"]
    
    # Customer segmentation
    bins = [0, 1000, 2500, 5000, np.inf]
    labels = ["Low", "Avg", "High", "VIP"]
    basket_sum["segment"] = pd.cut(basket_sum["total_sales"], bins=bins, labels=labels)
    
    # Frequency segmentation
    basket_sum["frequency_segment"] = pd.qcut(basket_sum["basket_count"], 
                                              q=4, 
                                              labels=["Rare", "Occasional", "Frequent", "Very Frequent"])
    
    basket_sum["overall_segment"] = basket_sum["segment"].astype(str) + ' - ' + basket_sum["frequency_segment"].astype(str)
    
    # Segment profile
    segment_profile_df = pd.merge(basket_sum, hh_demographic, on="household_id", how="inner")
    
    print("Saving processed datasets...")
    
    datasets_to_save = {
        "campaign_desc_processed.csv": camp_desc,
        "campaign_table_processed.csv": camp_table,
        "causal_data_processed.csv": causal_data,
        "coupon_redempt_processed.csv": coupon_redempt,
        "coupon_processed.csv": coupon,
        "hh_demographic_processed.csv": hh_demographic,
        "product_processed.csv": product,
        "transaction_data_processed.csv": transaction,
        "basket_fact.csv": basket_fact,
        "basket_sum.csv": basket_sum,
        "segment_profile.csv": segment_profile_df
    }
    
    for filename, df in datasets_to_save.items():
        filepath = os.path.join(processed_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filename} ({len(df):,} rows, {len(df.columns)} columns)")
    
    print(f"\nAll processed datasets saved to: {processed_dir}")
    
    # Print summary statistics
    print("\nDataset Summary:")
    for filename, df in datasets_to_save.items():
        print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns")

if __name__ == "__main__":
    process_and_save_datasets()