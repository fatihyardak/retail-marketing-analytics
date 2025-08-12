import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Dosya yolu kontrolü
file_path = "../datasets/processed/customer_features.csv"
if not os.path.exists(file_path):
    print(f"not found: {file_path}")
    

customer_features_df = pd.read_csv(file_path)
print(f"data loaded: {customer_features_df.shape[0]} row, {customer_features_df.shape[1]} column")

demographic_cols = ["age_desc", "marital_status_code", "income_desc", 
                    "homeowner_desc", "hh_comp_desc", "household_size_desc", 
                    "kid_category_desc"]

for col in demographic_cols:
    if col in customer_features_df.columns:
        customer_features_df[col] = customer_features_df[col].fillna("Unknown", inplace=True)

# one hot encoding
categorical_cols = customer_features_df.select_dtypes(include=["object", "category"]).columns.drop(["segment", "frequency_segment", "overall_segment"])

customer_features_encoded = pd.get_dummies(customer_features_df, columns=categorical_cols, drop_first=True)

# feature scaling
num_cols = ["total_sales", "basket_count", "avg_basket_size", "days_since_last_purchase", "total_discount", 
                  "total_coupons_redeemed", "discount_ratio"]

num_cols2 = []
for col in num_cols:
    if col in customer_features_encoded.columns:
        num_cols2.append(col)

scaler = StandardScaler()
scaled_num_cols = scaler.fit_transform(customer_features_encoded[num_cols2])

scaled_numerical_df = pd.DataFrame(scaled_num_cols, columns=num_cols2, index=customer_features_encoded.index)

model_ready_df = customer_features_encoded.drop(columns=num_cols2)
model_ready_df = pd.concat([model_ready_df, scaled_numerical_df], axis=1)

model_ready_df.set_index("household_id", inplace=True)


model_ready_df.drop(columns=["segment", "frequency_segment", "overall_segment"], inplace=True)

print(model_ready_df.head())

if __name__ == "__main__":

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "../datasets/processed/model_ready_features.csv")
    output_path = os.path.normpath(output_path)
    model_ready_df.to_csv(output_path)
    print(f"\nModel ready data '{output_path}' saved with success")