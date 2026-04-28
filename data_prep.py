#s16798
# data_prep.py
#data_prep.py defines a function that loads a car dataset and cleans it before analysis

#import python library
import pandas as pd

#Define the function
def load_and_clean_data(csv_path):

    #Load CSV file from the s16798 folder
    df = pd.read_csv(r"car_price_dataset .csv")

    #Remove unnecessary column
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    #  Basic data Cleaning steps

    #Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    #Create a new variable (Age)
    df["Age"] = 2026 - df["YOM"]

    #Convert car features into binary variables
    for col in ["AIR CONDITION", "POWER STEERING", "POWER MIRROR", "POWER WINDOW"]:
        df[col + "_bin"] = (df[col] == "Available").astype(int)

    df["Leasing_bin"]   = (df["Leasing"] != "No Leasing").astype(int)
    df["Gear_bin"]      = (df["Gear"] == "Automatic").astype(int)
    df["Condition_bin"] = (df["Condition"] == "NEW").astype(int)

    # Strip strings(Clean text columns)
    df["Brand"] = df["Brand"].astype(str).str.strip().str.title()
    df["Model"] = df["Model"].astype(str).str.strip().str.title()
    df["Town"]  = df["Town"].astype(str).str.strip().str.title()

    #Remove empty brand/model
    df = df[(df["Brand"] != "") & (df["Model"] != "")]

    # Drop duplicates / missing
    df_clean = df.dropna().drop_duplicates().copy()

    # ── Town → Province Mapping (Sri Lanka) ──
    town_to_province = {
        # Western Province
        "Colombo": "Western", "Gampaha": "Western", "Negombo": "Western",
        "Kalutara": "Western", "Panadura": "Western", "Moratuwa": "Western",
        "Dehiwala-Mount-Lavinia": "Western", "Maharagama": "Western",
        "Kotte": "Western", "Wattala": "Western", "Ja-Ela": "Western",
        "Kelaniya": "Western", "Kadawatha": "Western", "Nugegoda": "Western",
        "Piliyandala": "Western", "Boralesgamuwa": "Western",
        # Central Province
        "Kandy": "Central", "Matale": "Central", "Nuwara-Eliya": "Central",
        "Gampola": "Central", "Nawalapitiya": "Central", "Hatton": "Central",
        # Southern Province
        "Galle": "Southern", "Matara": "Southern", "Hambantota": "Southern",
        "Weligama": "Southern", "Tangalle": "Southern", "Hikkaduwa": "Southern",
        "Ambalangoda": "Southern",
        # Northern Province
        "Jaffna": "Northern", "Vavuniya": "Northern", "Kilinochchi": "Northern",
        "Mullaitivu": "Northern",
        # Eastern Province
        "Batticaloa": "Eastern", "Trincomalee": "Eastern", "Ampara": "Eastern",
        "Kalmunai": "Eastern",
        # North Western Province
        "Kurunegala": "North Western", "Puttalam": "North Western",
        "Kuliyapitiya": "North Western", "Chilaw": "North Western",
        # North Central Province
        "Anuradapura": "North Central", "Polonnaruwa": "North Central",
        # Uva Province
        "Badulla": "Uva", "Bandarawela": "Uva", "Haputale": "Uva", "Welimada": "Uva",
        # Sabaragamuwa Province
        "Ratnapura": "Sabaragamuwa", "Kegalle": "Sabaragamuwa", "Balangoda": "Sabaragamuwa"
    }

    # Map Town → Province
    df["Province"] = df["Town"].map(town_to_province)
    df["Province"] = df["Province"].fillna("Other")

    #Return the results

    return df, df_clean