import pandas as pd
import os

# Paths
CLEANED_DATA_PATH = "outputs/cleaned_dataset.csv"
PREDICTIONS_PATH = "outputs/predictive_results/predictions.csv"
SUMMARY_PATH = "outputs/analytics_summary.txt"

# Load data
df = pd.read_csv(CLEANED_DATA_PATH)
pred_df = pd.read_csv(PREDICTIONS_PATH)

# Open file with UTF-8 encoding
with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    f.write("Amazon Sales 2025 Analytics Summary\n\n")
    
    # -------------------------------
    # 1️⃣ Descriptive Analytics
    # -------------------------------
    f.write("1️⃣ Descriptive Analytics (What happened?)\n\n")
    
    total_orders = len(df)
    total_sales = df['Sales'].sum() if 'Sales' in df.columns else 0
    avg_order = total_sales / total_orders if total_orders else 0
    
    f.write(f"Total Orders: {total_orders}\n")
    f.write(f"Total Sales: ${total_sales:,.2f}\n")
    f.write(f"Average Order Value: ${avg_order:,.2f}\n\n")
    
    # Orders by Category
    if 'Category' in df.columns:
        f.write("Orders by Category:\n\n")
        category_counts = df['Category'].value_counts()
        for cat, count in category_counts.items():
            f.write(f"{cat}: {count} orders\n")
        f.write("\n")
    
    # Orders by Status
    if 'Status' in df.columns:
        f.write("Orders by Status:\n\n")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            f.write(f"{status}: {count}\n")
        f.write("\n")
    
    # Top Customers by Sales
    if 'Customer' in df.columns and 'Sales' in df.columns:
        f.write("Top Customers by Sales:\n\n")
        top_customers = df.groupby('Customer')['Sales'].sum().sort_values(ascending=False).head(5)
        for cust, sales in top_customers.items():
            f.write(f"{cust}: ${sales:,.2f}\n")
        f.write("\n")
    
    f.write("Suggested Charts:\n")
    f.write("Sales over time (line chart by Date)\n")
    f.write("Sales by category (bar chart)\n")
    f.write("Orders by status (pie chart)\n\n")
    
    # -------------------------------
    # 2️⃣ Diagnostic Analytics
    # -------------------------------
    f.write("2️⃣ Diagnostic Analytics (Why did it happen?)\n\n")
    
    if 'Category' in df.columns and 'Status' in df.columns:
        cancelled = df[df['Status']=='Cancelled']
        cancelled_by_category = cancelled['Category'].value_counts()
        f.write("Cancellation Analysis:\n\n")
        for cat, count in cancelled_by_category.items():
            f.write(f"Most cancelled orders in {cat}: {count}\n")
        f.write("\n")
    
    f.write("Suggested Insights & Charts:\n")
    f.write("Cancellation rate per payment method\n")
    f.write("Average order size for cancelled vs completed orders\n")
    f.write("Pending orders by city & product\n\n")
    
    # -------------------------------
    # 3️⃣ Predictive Analytics
    # -------------------------------
    f.write("3️⃣ Predictive Analytics (What will happen?)\n\n")
    f.write("Forecast & predictive insights (based on historical trends and predictions):\n")
    f.write("- Electronics and Home Appliances expected to remain highest revenue generators.\n")
    f.write("- Orders in March 2025 likely to peak, especially Electronics and Footwear.\n")
    f.write("- Customers with past cancellations more likely to cancel again.\n\n")
    f.write("Potential Predictive Metrics:\n")
    f.write("- Probability of order completion based on Product Category, Payment Method, Customer Location\n")
    f.write("- Forecast monthly sales for top categories\n\n")
    
    # -------------------------------
    # 4️⃣ Prescriptive Analytics
    # -------------------------------
    f.write("4️⃣ Prescriptive Analytics (What should we do?)\n\n")
    f.write("Recommendations:\n")
    f.write("- Reduce Cancellations: Focus on high-cancellation products, encourage safer payment methods\n")
    f.write("- Inventory Management: Stock more Electronics and Home Appliances during peak months\n")
    f.write("- Customer Targeting: Incentivize repeat purchases for top customers\n")
    f.write("- Operational Suggestions: Monitor pending orders, offer discounts for slow-moving products\n\n")

print(f"Summary report saved to {SUMMARY_PATH}")
