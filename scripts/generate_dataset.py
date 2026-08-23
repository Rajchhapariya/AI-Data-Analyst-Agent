"""
Dataset Generator: Global Superstore Retail Sales
Generates a realistic, multi-dimensional sales transactions dataset for the AI Data Analyst Agent.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Set deterministic seed for reproducibility
random.seed(42)
np.random.seed(42)

REGIONS_STATES_CITIES = {
    "West": {
        "California": ["Los Angeles", "San Francisco", "San Diego", "San Jose", "Sacramento"],
        "Washington": ["Seattle", "Spokane", "Tacoma", "Bellevue"],
        "Oregon": ["Portland", "Eugene", "Salem"],
        "Arizona": ["Phoenix", "Tucson", "Mesa"],
        "Colorado": ["Denver", "Colorado Springs", "Aurora"]
    },
    "East": {
        "New York": ["New York City", "Buffalo", "Rochester", "Albany"],
        "Pennsylvania": ["Philadelphia", "Pittsburgh", "Allentown"],
        "Massachusetts": ["Boston", "Worcester", "Springfield"],
        "New Jersey": ["Newark", "Jersey City", "Paterson"],
        "Ohio": ["Columbus", "Cleveland", "Cincinnati"]
    },
    "Central": {
        "Illinois": ["Chicago", "Aurora", "Naperville", "Rockford"],
        "Texas": ["Houston", "San Antonio", "Dallas", "Austin", "Fort Worth"],
        "Michigan": ["Detroit", "Grand Rapids", "Warren"],
        "Minnesota": ["Minneapolis", "Saint Paul", "Rochester"],
        "Indiana": ["Indianapolis", "Fort Wayne", "Evansville"]
    },
    "South": {
        "Florida": ["Jacksonville", "Miami", "Tampa", "Orlando", "St. Petersburg"],
        "North Carolina": ["Charlotte", "Raleigh", "Greensboro"],
        "Georgia": ["Atlanta", "Augusta", "Columbus", "Savannah"],
        "Virginia": ["Virginia Beach", "Norfolk", "Chesapeake", "Richmond"],
        "Tennessee": ["Nashville", "Memphis", "Knoxville"]
    }
}

CAT_HIERARCHY = {
    "Furniture": {
        "Bookcases": ["Bush Somerset Collection Bookcase", "Sauder Multifunction Bookcase", "Riverside Palais Royal Bookcase", "O'Sullivan 2-Door Bookcase"],
        "Chairs": ["Hon Deluxe Fabric Mid-Back Chair", "Novimex Swivel Mesh Chair", "Harbour Creations Executive Leather Armchair", "Global High-Back Executive Leather Chair"],
        "Furnishings": ["Eldon Expressions Desk Accessories", "Rubbermaid Cluster Cylinder Wastebasket", "Howard Miller 13-3/4\" Diameter Wall Clock", "Tensor Traditional Floor Lamp"],
        "Tables": ["Chromcraft Round Conference Table", "Bevis Rectangular Conference Table", "Lesro Round Coffee Table", "Balt Training Tables with Wire Management"]
    },
    "Office Supplies": {
        "Appliances": ["Holmes Replacement Filter", "Avanti 4.4 Cu. Ft. Refrigerator", "Kensington Orbit Trackball", "Fellowes Powershred Cross-Cut Shredder"],
        "Art": ["Newell 312 Series Ballpoint Pens", "Sanford Liquid Accent Highlighters", "Prang Colored Pencils 24ct", "BIC Soft Feel Retractable Ball Pens"],
        "Binders": ["Avery Heavy-Duty Ring Binders", "Wilson Jones Hanging Data Binders", "GBC VeloBind System Covers", "Fellowes Heavy-Duty Storage Box"],
        "Envelopes": ["Staples #10 Business Envelopes", "Quality Park Clasp Envelopes", "Tyvek Heavy-Duty Self-Sealing Mailers"],
        "Fasteners": ["Advantus Map Tacks", "Acco Heavy-Duty Brass Fasteners", "Ideal Clamp Binder Clips"],
        "Labels": ["Avery 5160 Laser Address Labels", "Avery Printable File Folder Labels", "Dymo Direct Thermal Shipping Labels"],
        "Paper": ["Xerox 196 Universal Multipurpose Paper", "Hammermill Premium Color Copy Paper", "Southworth 25% Cotton Resume Paper"],
        "Storage": ["Sterilite Modular Storage Drawers", "Fellowes Plastic File Storage Box", "Akro-Mils 24-Drawer Plastic Cabinet"],
        "Supplies": ["Acme Stainless Steel Scissors", "Fiskars Titanium Softgrip Scissors", "Martin Yale Industrial Paper Trimmer"]
    },
    "Technology": {
        "Accessories": ["Logitech Wireless Marathon Mouse", "SanDisk 64GB Ultra USB 3.0 Flash Drive", "Anker 10-Port USB Data Hub", "Plantronics Voyager Bluetooth Headset"],
        "Copiers": ["Canon ImageCLASS Digital Multifunction Copier", "Hewlett Packard LaserJet Enterprise Copier", "Sharp AL-1530CS Digital Copier", "Brother Personal Fax & Copier"],
        "Machines": ["Epson WorkForce Pro Wireless Color All-in-One", "Zebra Thermal Label Printer", "Dymo LabelWriter 450 Turbo", "Star Micronics Impact POS Receipt Printer"],
        "Phones": ["Apple iPhone 13 Pro 128GB", "Samsung Galaxy S22 Ultra", "Cisco Unified IP Phone 7965G", "Motorola Moto G Stylus"]
    }
}

CUSTOMER_NAMES = [
    "Claire Gute", "Brosina Hoffman", "Darrin Van Huff", "Sean O'Donnell", "Zuschuss Donatelli",
    "Ken Black", "Sandra Flanagan", "Emily Phan", "Eric Hoffmann", "Tracy Blumstein",
    "Matt Abelman", "Gene Hale", "Steve Nguyen", "Linda Cazamias", "Ruben Dartt",
    "Erin Smith", "Patrick O'Donnell", "Harold Pawlan", "Pete Kriz", "Shirley Schmidt",
    "Dave Brooks", "Arthur Prichep", "Paul Prost", "Jack O'Briant", "Valerie Mitchum",
    "Alejandro Ballentine", "Camille Belle", "Don Miller", "Lena Cacioppo", "Liz Pelletier",
    "Maria Etezadi", "Mitch Willingham", "Nora Pelletier", "Quincy Jones", "Rachel Tyler",
    "Sanjit Chand", "Toby Braunhardt", "Victor Vance", "William Brown", "Yana Sorensen"
]

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SEGMENT_WEIGHTS = [0.52, 0.30, 0.18]

SHIP_MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]
SHIP_WEIGHTS = [0.60, 0.20, 0.15, 0.05]

PRIORITIES = ["Low", "Medium", "High", "Critical"]
PRIORITY_WEIGHTS = [0.15, 0.55, 0.20, 0.10]

def generate_superstore_dataset(num_records: int = 7500, output_path: str = "data/superstore_sales.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    records = []
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2024, 12, 31)
    total_days = (end_date - start_date).days

    for i in range(1, num_records + 1):
        order_day_offset = random.randint(0, total_days)
        order_date = start_date + timedelta(days=order_day_offset)
        
        ship_mode = random.choices(SHIP_MODES, weights=SHIP_WEIGHTS)[0]
        days_to_ship = {
            "Same Day": 0,
            "First Class": random.randint(1, 2),
            "Second Class": random.randint(2, 4),
            "Standard Class": random.randint(3, 7)
        }[ship_mode]
        ship_date = order_date + timedelta(days=days_to_ship)
        
        region = random.choice(list(REGIONS_STATES_CITIES.keys()))
        state = random.choice(list(REGIONS_STATES_CITIES[region].keys()))
        city = random.choice(REGIONS_STATES_CITIES[region][state])
        
        customer_name = random.choice(CUSTOMER_NAMES)
        customer_id = f"{customer_name[:2].upper()}-{hash(customer_name) % 90000 + 10000}"
        segment = random.choices(CUSTOMER_SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        
        category = random.choice(list(CAT_HIERARCHY.keys()))
        sub_category = random.choice(list(CAT_HIERARCHY[category].keys()))
        product_name = random.choice(CAT_HIERARCHY[category][sub_category])
        product_id = f"{category[:3].upper()}-{sub_category[:2].upper()}-{hash(product_name) % 90000000 + 10000000}"
        
        # Base pricing dynamics
        if category == "Technology":
            base_unit_price = random.uniform(80.0, 1800.0) if sub_category in ["Copiers", "Phones", "Machines"] else random.uniform(25.0, 180.0)
            margin_rate = random.uniform(0.15, 0.45)
        elif category == "Furniture":
            base_unit_price = random.uniform(90.0, 950.0) if sub_category in ["Chairs", "Tables", "Bookcases"] else random.uniform(15.0, 120.0)
            # Tables and Bookcases often have thin or negative margins due to heavy shipping/discount
            margin_rate = random.uniform(-0.25, 0.20) if sub_category in ["Tables", "Bookcases"] else random.uniform(0.05, 0.30)
        else: # Office Supplies
            base_unit_price = random.uniform(3.0, 85.0) if sub_category not in ["Appliances", "Storage"] else random.uniform(40.0, 350.0)
            margin_rate = random.uniform(0.10, 0.50)
            
        quantity = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], weights=[0.20, 0.22, 0.18, 0.14, 0.10, 0.06, 0.04, 0.03, 0.015, 0.01, 0.005])[0]
        
        # Discounts
        discount = random.choices([0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8], weights=[0.48, 0.15, 0.10, 0.12, 0.05, 0.04, 0.03, 0.02, 0.01])[0]
        
        gross_sales = base_unit_price * quantity
        sales = round(gross_sales * (1.0 - discount), 2)
        
        # Profit depends on margin rate and discount impact
        cost = gross_sales * (1.0 - margin_rate)
        profit = round(sales - cost, 2)
        
        # Shipping cost proportional to weight/mode/sales
        mode_multiplier = {"Same Day": 1.8, "First Class": 1.4, "Second Class": 1.1, "Standard Class": 0.8}[ship_mode]
        shipping_cost = round(max(2.5, (sales * 0.08 + quantity * 2.2) * mode_multiplier), 2)
        
        priority = random.choices(PRIORITIES, weights=PRIORITY_WEIGHTS)[0]
        order_id = f"CA-{order_date.year}-{100000 + i}"
        
        # Realistic data nuances: small missing values (~0.5% in postal/shipping or discount)
        # to ensure data profiler accurately discovers them
        record = {
            "order_id": order_id,
            "order_date": order_date.strftime("%Y-%m-%d"),
            "ship_date": ship_date.strftime("%Y-%m-%d"),
            "ship_mode": ship_mode,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "segment": segment,
            "city": city,
            "state": state,
            "region": region,
            "category": category,
            "sub_category": sub_category,
            "product_id": product_id,
            "product_name": product_name,
            "sales": sales,
            "quantity": quantity,
            "discount": discount if random.random() > 0.005 else np.nan, # 0.5% null discount
            "profit": profit,
            "shipping_cost": shipping_cost if random.random() > 0.008 else np.nan, # 0.8% null shipping
            "order_priority": priority
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print(f"Schema columns: {list(df.columns)}")
    print(f"Total Sales: ${df['sales'].sum():,.2f}, Total Profit: ${df['profit'].sum():,.2f}")
    return df

if __name__ == "__main__":
    generate_superstore_dataset()
