import streamlit as st
import joblib
import sklearn



sc = joblib.load("scalingtype.sav")



model = joblib.load('bigmartmodel.sav')


def main():
    st.title("BigMart Sales Prediction")
    html_temp = """  
       <h4>References for Item Type</h4>
       <table>
    <tr>
        <th>Value</th>
        <th>Item Type</th>
    </tr>
    <tr>
        <td>0</td>
        <td>Baking Goods</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Breads</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Breakfast</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Canned</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Dairy</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Frozen Foods</td>
    </tr>
    <tr>
        <td>6</td>
        <td>Fruits and Vegetables</td>
    </tr>
    <tr>
        <td>7</td>
        <td>Hard Drinks</td>
    </tr>
    <tr>
        <td>8</td>
        <td>Health and Hygiene</td>
    </tr>
    <tr>
        <td>9</td>
        <td>Household</td>
    </tr>
    <tr>
        <td>10</td>
        <td>Meat</td>
    </tr>
    <tr>
        <td>11</td>
        <td>Others</td>
    </tr>
    <tr>
        <td>12</td>
        <td>Seafood</td>
    </tr>
    <tr>
        <td>13</td>
        <td>Snack Foods</td>
    </tr>
    <tr>
        <td>14</td>
        <td>Soft Drinks</td>
    </tr>
    <tr>
        <td>15</td>
        <td>Strachy Foods</td>
    </tr>
    
</table>
       """
    st.markdown(html_temp, unsafe_allow_html=True)
    # Set the title and header of the Streamlit app
    st.header("Enter the input details for prediction")

    # Collect user input details
    item_weight = st.number_input("Item Weight",value=None)
    item_fat_content = st.selectbox("Item Fat Content(0 for Low Fat and 1 for Regular)", ["0", "1"])
    item_visibility = st.number_input("Item Visibility",value=None)
    item_type = st.selectbox("Item Type",["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"])
    item_mrp = st.number_input("Item MRP",value=None)
    outlet_establishment_year = st.selectbox("Outlet Establishment Year",
                                             ["1985", "1987", "1997", "1998", "1999", "2002", "2004", "2007", "2009"])
    outlet_size=st.selectbox("Outlet Size(0 for High,1 for Medium and 2 for Small)", ["0", "1", "2"])
    outlet_location_type=st.selectbox("Outlet Location Type(0 for Tier 1, 1-Tier 2 and 2 for Tier 3)", ["0", "1", "2"])
    outlet_type=st.selectbox("Outlet Type(0 for Grocery Store,1 for Supermarket Type1,2 for Supermarket Type2,3 for Supermarket3",["0","1","2","3"])

    if st.button("Predict"):
        user_input = [[item_weight, item_fat_content, item_visibility, item_type, item_mrp, outlet_establishment_year,
                      outlet_size, outlet_location_type, outlet_type]]

        X_std = sc.transform(user_input)

        prediction = model.predict(X_std)

        st.write(f"The predicted sales for this item is: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()