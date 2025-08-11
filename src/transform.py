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

def create_customer_features(transaction, hh_demographic, coupon_redempt):
    """Create comprehensive customer features including RFM analysis and discount behavior"""
    
    
    t = transaction[["household_id", "basket_id", "day", "sales_value"]].copy()
    basket_sum = t.groupby("household_id").agg(
        total_sales=("sales_value", "sum"),
        basket_count=("basket_id", "nunique")
    ).reset_index()
    
    basket_sum["avg_basket_size"] = basket_sum["total_sales"] / basket_sum["basket_count"]
    
    # customer segmentation based total sales
    bins = [0, 1000, 2500, 5000, np.inf]
    labels = ["Low", "Avg", "High", "VIP"]
    basket_sum["segment"] = pd.cut(basket_sum["total_sales"], bins=bins, labels=labels)
    
    # freq segmentation
    basket_sum["frequency_segment"] = pd.qcut(basket_sum["basket_count"], 
                                              q=4, 
                                              labels=["Rare", "Occasional", "Frequent", "Very Frequent"])
    
    basket_sum["overall_segment"] = basket_sum["segment"].astype(str) + ' - ' + basket_sum["frequency_segment"].astype(str)
    
    # recency 
    analysis_day = transaction["day"].max()
    last_purchase_df = transaction.groupby("household_id")["day"].max().reset_index()
    last_purchase_df.rename(columns={"day": "last_purchase_day"}, inplace=True)
    last_purchase_df["recency"] = analysis_day - last_purchase_df["last_purchase_day"]
    
    # merging recency with basket summary
    basket_sum = pd.merge(basket_sum, last_purchase_df[['household_id', 'recency']], on='household_id', how='left')
    
    # discount
    discount_features = transaction.groupby("household_id").agg({
        "retail_disc": lambda x: x.abs().sum(),
        "coupon_disc": lambda x: x.abs().sum()
    }).reset_index()
    
    discount_features.rename(columns={
        "retail_disc": "total_retail_disc", 
        "coupon_disc": "total_coupon_disc"
    }, inplace=True)
    
    discount_features["total_discount"] = discount_features["total_retail_disc"] + discount_features["total_coupon_disc"]
    
    # coupon
    coupon_usage = coupon_redempt.groupby("household_id").size().reset_index(name="total_coupons_redeemed")
    
    # merging 
    customer_features = pd.merge(basket_sum, discount_features, on="household_id", how="left")
    customer_features = pd.merge(customer_features, coupon_usage, on="household_id", how="left")
    
    # missing values
    customer_features["total_discount"] = customer_features["total_discount"].fillna(0)
    customer_features["total_coupons_redeemed"] = customer_features["total_coupons_redeemed"].fillna(0)
    
    # discount ratio
    customer_features["discount_ratio"] = (customer_features["total_discount"] / customer_features["total_sales"]).fillna(0)
    
    # Adding demogrphic info 
    customer_features = pd.merge(customer_features, hh_demographic, on="household_id", how="left")
    
    return customer_features

def create_product_analysis(transaction, product, basket_sum):
    """Create product analysis with customer segments"""
    
    # Merge transaction data with customer segments
    trans_with_segment = pd.merge(transaction, basket_sum[["household_id", "segment"]], on="household_id", how='left')
    
    # Full product analysis dataset
    full_df = pd.merge(trans_with_segment, product, on="product_id", how="left")
    
    # Product performance by segment
    product_segment_performance = full_df.groupby(["department", "segment"])["sales_value"].sum().reset_index()
    
    # Top products by segment
    top_products_by_segment = full_df.groupby(["product_id", "segment"])["sales_value"].sum().reset_index()
    
    return full_df, product_segment_performance, top_products_by_segment

def process_datasets():
    """Process raw datasets and save to processed folder"""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    raw_dir = os.path.join(project_root, "datasets", "raw")
    processed_dir = os.path.join(project_root, "datasets", "processed")
    
    os.makedirs(processed_dir, exist_ok=True)
    
    print(f"Reading from: {raw_dir}")
    print(f"Saving to: {processed_dir}")
    
    # Loading raw datasets
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
    
    # Creatingg comprehensive customer features
    customer_features = create_customer_features(transaction, hh_demographic, coupon_redempt)
    
    # Create product analysis
    full_product_df, product_segment_performance, top_products_by_segment = create_product_analysis(
        transaction, product, customer_features[["household_id", "segment"]]
    )
    
    # Segment profile 
    segment_profile_df = pd.merge(
        customer_features[["household_id", "segment", "frequency_segment", "overall_segment"]], 
        hh_demographic, on="household_id", how="inner"
    )
    
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
        "customer_features.csv": customer_features,
        "segment_profile.csv": segment_profile_df,
        "product_segment_performance.csv": product_segment_performance,
        "top_products_by_segment.csv": top_products_by_segment
    }
    
    for filename, df in datasets_to_save.items():
        filepath = os.path.join(processed_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved: {filename} ({len(df):,} rows, {len(df.columns)} columns)")
    
    print(f"\nAll processed datasets saved to: {processed_dir}")
    
    # Summary stats
    print("\nDataset Summary:")
    for filename, df in datasets_to_save.items():
        print(f"{filename}: {len(df):,} rows, {len(df.columns)} columns")
    
    # Customer segmentation summary
    print("\nCustomer Segmentation Summary:")
    if "segment" in customer_features.columns:
        segment_counts = customer_features["segment"].value_counts()
        print("Sales-based segments:")
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count:,} customers")
    
    if "frequency_segment" in customer_features.columns:
        freq_counts = customer_features["frequency_segment"].value_counts()
        print("\nFrequency-based segments:")
        for segment, count in freq_counts.items():
            print(f"  {segment}: {count:,} customers")

if __name__ == "__main__":
    process_datasets()