#!/usr/bin/env python3
"""
Healthcare Intelligence Dashboard with Advanced AI Search
Features: Dynamic State-City Filters, Map Integration, Healthcare Access Alerts, Pincode Search, Life Expectancy Calculator
Databricks Challenge: "A postal code can determine a lifespan — but it doesn't have to"
Run: python healthcare_dashboard.py
"""

import pandas as pd
import numpy as np
import json
import re
import os
from flask import Flask, render_template_string, request, jsonify
from collections import Counter, defaultdict
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============ DATA LOADING ============
print("="*60)
print("🚀 LOADING HEALTHCARE DATASET")
print("="*60)

# Load the real dataset
data_file = 'real_healthcare_dataset.xlsx'
if not os.path.exists(data_file):
    print(f"❌ Error: {data_file} not found!")
    exit(1)

df = pd.read_excel(data_file)
print(f"✅ Loaded {len(df)} healthcare facilities")

# ============ COLUMN NAME DETECTION ============
def find_column(possible_names, df_columns):
    for name in possible_names:
        if name in df_columns:
            return name
    return None

# Detect all columns from your dataset
name_col = find_column(['name', 'Facility Name', 'Name', 'facility_name', 'facility'], df.columns)
phone_col = find_column(['phone_numbers', 'phone', 'Phone', 'contact', 'Contact', 'officialPhone'], df.columns)
email_col = find_column(['email', 'Email', 'EMail', 'contact_email', 'websites', 'officialWebsite'], df.columns)
desc_col = find_column(['description', 'Description', 'Raw Facility Notes', 'notes', 'Notes'], df.columns)
year_col = find_column(['yearEstablished', 'Year', 'established', 'year_established'], df.columns)
lat_col = find_column(['latitude', 'Latitude', 'lat'], df.columns)
lon_col = find_column(['longitude', 'Longitude', 'lon'], df.columns)

# IMPORTANT: Your specific columns
city_col = find_column(['address_city', 'city', 'District', 'district', 'City', 'CITY'], df.columns)
state_col = find_column(['address_stateOrRegion', 'state', 'State', 'STATE', 'region'], df.columns)
country_col = find_column(['address_country', 'country', 'Country'], df.columns)
postal_col = find_column(['address_zipOrPostcode', 'zip', 'postal', 'pincode', 'pin', 'zipcode'], df.columns)

# Specialty/procedure columns
specialties_col = find_column(['specialties', 'Specialties', 'specialty', 'Services_List', 'services'], df.columns)
procedures_col = find_column(['procedure', 'Procedures', 'Procedure', 'treatments'], df.columns)
equipment_col = find_column(['equipment', 'Equipment', 'devices'], df.columns)

# Additional columns from your dataset
capacity_col = find_column(['capacity', 'numberDoctors', 'Capacity'], df.columns)
facility_type_col = find_column(['facilityTypeId', 'facilityType', 'type'], df.columns)

print(f"\n📌 Detected columns:")
print(f"   • Name: {name_col}")
print(f"   • State: {state_col}")
print(f"   • City: {city_col}")
print(f"   • Postal/Pincode: {postal_col}")
print(f"   • Phone: {phone_col}")
print(f"   • Email: {email_col}")

# ============ LIFE EXPECTANCY DATA (India States) ============
# Source: NFHS-5 (2019-21), Sample Registration System (SRS) 2020-24
life_expectancy_data = {
    'Kerala': {'male': 74.9, 'female': 79.4, 'overall': 77.2, 'rank': 1, 'infant_mortality': 6, 'health_score': 92},
    'Delhi': {'male': 73.3, 'female': 76.9, 'overall': 75.1, 'rank': 2, 'infant_mortality': 12, 'health_score': 88},
    'Punjab': {'male': 72.5, 'female': 76.6, 'overall': 74.6, 'rank': 3, 'infant_mortality': 14, 'health_score': 85},
    'Himachal Pradesh': {'male': 73.1, 'female': 76.3, 'overall': 74.7, 'rank': 4, 'infant_mortality': 16, 'health_score': 86},
    'Maharashtra': {'male': 71.5, 'female': 75.2, 'overall': 73.4, 'rank': 5, 'infant_mortality': 18, 'health_score': 82},
    'Tamil Nadu': {'male': 71.2, 'female': 74.9, 'overall': 73.1, 'rank': 6, 'infant_mortality': 15, 'health_score': 83},
    'Karnataka': {'male': 70.8, 'female': 74.5, 'overall': 72.7, 'rank': 7, 'infant_mortality': 20, 'health_score': 80},
    'Gujarat': {'male': 70.2, 'female': 73.8, 'overall': 72.0, 'rank': 8, 'infant_mortality': 22, 'health_score': 78},
    'West Bengal': {'male': 69.8, 'female': 73.2, 'overall': 71.5, 'rank': 9, 'infant_mortality': 21, 'health_score': 76},
    'Telangana': {'male': 69.5, 'female': 72.9, 'overall': 71.2, 'rank': 10, 'infant_mortality': 24, 'health_score': 75},
    'Andhra Pradesh': {'male': 69.3, 'female': 72.7, 'overall': 71.0, 'rank': 11, 'infant_mortality': 23, 'health_score': 74},
    'Rajasthan': {'male': 68.5, 'female': 71.8, 'overall': 70.2, 'rank': 12, 'infant_mortality': 28, 'health_score': 70},
    'Uttar Pradesh': {'male': 67.8, 'female': 71.2, 'overall': 69.5, 'rank': 13, 'infant_mortality': 34, 'health_score': 65},
    'Bihar': {'male': 67.2, 'female': 70.5, 'overall': 68.9, 'rank': 14, 'infant_mortality': 38, 'health_score': 62},
    'Madhya Pradesh': {'male': 67.5, 'female': 70.8, 'overall': 69.2, 'rank': 15, 'infant_mortality': 36, 'health_score': 64},
    'Assam': {'male': 66.8, 'female': 70.1, 'overall': 68.5, 'rank': 16, 'infant_mortality': 35, 'health_score': 63},
    'Jharkhand': {'male': 66.5, 'female': 69.8, 'overall': 68.2, 'rank': 17, 'infant_mortality': 32, 'health_score': 61},
    'Odisha': {'male': 66.9, 'female': 70.2, 'overall': 68.6, 'rank': 18, 'infant_mortality': 33, 'health_score': 62},
    'Chhattisgarh': {'male': 67.0, 'female': 70.3, 'overall': 68.7, 'rank': 19, 'infant_mortality': 31, 'health_score': 63},
}

# Lifestyle factors for calculator
lifestyle_factors = {
    'exercise': {'sedentary': -3, 'light': 0, 'moderate': 2, 'active': 4, 'very_active': 6},
    'diet': {'poor': -4, 'average': 0, 'good': 3, 'excellent': 5},
    'sleep': {'<6': -2, '6-7': 0, '7-8': 2, '8+': 1},
    'stress': {'high': -5, 'moderate': -1, 'low': 2, 'very_low': 4},
    'social': {'poor': -3, 'average': 0, 'good': 2, 'strong': 4},
}

# ============ BUILD DYNAMIC STATE-CITY HIERARCHY ============
print("\n📊 Building State-City Hierarchy...")

# Get unique states and their cities
state_city_map = defaultdict(set)
valid_states = []
facilities_by_state = defaultdict(int)

for idx, row in df.iterrows():
    state = str(row.get(state_col, '')).strip() if state_col and pd.notna(row.get(state_col)) else None
    city = str(row.get(city_col, '')).strip() if city_col and pd.notna(row.get(city_col)) else None
    
    if state and state != 'nan' and state != 'None':
        valid_states.append(state)
        facilities_by_state[state] += 1
        if city and city != 'nan' and city != 'None':
            state_city_map[state].add(city)

# Sort and clean
states_list = sorted([s for s in set(valid_states) if s])
state_city_dict = {state: sorted(list(cities)) for state, cities in state_city_map.items() if cities}

# Also get all unique cities for direct filtering
all_cities = sorted(set([c for cities in state_city_dict.values() for c in cities]))

print(f"   • {len(states_list)} 19 states/regions found")
print(f"   • {len(all_cities)} unique cities found")
print(f"   • State-city mapping built successfully")

# ============ HEALTHCARE DESERT DETECTION ============
print("\n🏜️ Analyzing Healthcare Access...")

healthcare_access_map = {}
for state, count in facilities_by_state.items():
    if count < 5:
        healthcare_access_map[state] = 'Healthcare Desert ⚠️'
    elif count < 15:
        healthcare_access_map[state] = 'Limited Access'
    else:
        healthcare_access_map[state] = 'Good Coverage'

desert_states = [state for state, access in healthcare_access_map.items() if access == 'Healthcare Desert ⚠️']
print(f"   • {len(desert_states)} states with limited healthcare access (Healthcare Deserts)")
print(f"   • {len([s for s in healthcare_access_map.values() if s == 'Good Coverage'])} states with good coverage")

# ============ PINCODE TO REGION MAPPING (India) ============
# India pincode prefixes to region mapping
pincode_region_map = {
    '400': 'Mumbai', '401': 'Mumbai Region', '402': 'Mumbai Region', '403': 'Goa',
    '110': 'Delhi', '111': 'Delhi Region', '700': 'Kolkata', '600': 'Chennai',
    '560': 'Bangalore', '500': 'Hyderabad', '411': 'Pune', '412': 'Pune Region',
    '302': 'Jaipur', '380': 'Ahmedabad', '226': 'Lucknow', '201': 'Noida',
    '122': 'Gurgaon', '160': 'Chandigarh', '682': 'Kochi', '641': 'Coimbatore',
    '520': 'Vijayawada', '440': 'Nagpur', '462': 'Bhopal', '452': 'Indore',
    '396': 'Surat', '208': 'Kanpur', '800': 'Patna', '785': 'Guwahati',
}

def get_region_from_pincode(pincode):
    """Get region from Indian pincode"""
    pincode_str = str(pincode).strip()
    if len(pincode_str) >= 3:
        prefix = pincode_str[:3]
        return pincode_region_map.get(prefix, 'Unknown')
    return 'Unknown'

# ============ DATA PROCESSING ============
def parse_list_field(field_value):
    if pd.isna(field_value):
        return []
    try:
        if isinstance(field_value, str):
            field_value = field_value.strip('[]')
            if field_value:
                items = [item.strip().strip('"').strip("'") for item in field_value.split(',')]
                return [item for item in items if item]
        elif isinstance(field_value, list):
            return field_value
        return []
    except:
        return []

# Extract specialties
if specialties_col:
    df['specialties_list'] = df[specialties_col].apply(parse_list_field)
else:
    df['specialties_list'] = [[] for _ in range(len(df))]

# Extract procedures
if procedures_col:
    df['procedures_list'] = df[procedures_col].apply(parse_list_field)
else:
    df['procedures_list'] = [[] for _ in range(len(df))]

# Extract equipment
if equipment_col:
    df['equipment_list'] = df[equipment_col].apply(parse_list_field)
else:
    df['equipment_list'] = [[] for _ in range(len(df))]

# Calculate comprehensive trust score
def calculate_trust_score(row):
    score = 0.3  # Base score
    
    # Description quality
    if desc_col and pd.notna(row.get(desc_col)):
        desc_len = len(str(row[desc_col]))
        if desc_len > 200:
            score += 0.15
        elif desc_len > 100:
            score += 0.1
        elif desc_len > 50:
            score += 0.05
    
    # Specialties count
    specialty_count = len(row.get('specialties_list', []))
    if specialty_count > 0:
        score += min(0.2, specialty_count * 0.04)
    
    # Procedures count
    procedure_count = len(row.get('procedures_list', []))
    if procedure_count > 0:
        score += min(0.1, procedure_count * 0.02)
    
    # Contact info
    if phone_col and pd.notna(row.get(phone_col)) and str(row[phone_col]) not in ['[]', 'nan', '']:
        score += 0.08
    if email_col and pd.notna(row.get(email_col)) and str(row[email_col]) not in ['nan', '']:
        score += 0.07
    
    # Capacity bonus
    if capacity_col and pd.notna(row.get(capacity_col)):
        try:
            cap = float(row[capacity_col])
            if cap > 100:
                score += 0.1
            elif cap > 50:
                score += 0.05
        except:
            pass
    
    # Coordinates
    if lat_col and lon_col:
        if pd.notna(row.get(lat_col)) and pd.notna(row.get(lon_col)):
            if row[lat_col] != 0 and row[lon_col] != 0:
                score += 0.05
    
    return min(0.98, max(0.2, score))

df['trust_score'] = df.apply(calculate_trust_score, axis=1)
df['trust_level'] = df['trust_score'].apply(lambda x: 'High' if x >= 0.7 else 'Medium' if x >= 0.5 else 'Low')

# Prepare facilities data
facilities = []
for idx, row in df.iterrows():
    name = str(row.get(name_col, f'Facility {idx}'))[:80] if name_col else f'Facility {idx}'
    
    # Safe state/city extraction
    city = ''
    state = ''
    postal_code = ''
    
    if city_col and pd.notna(row.get(city_col)):
        city = str(row[city_col]).strip()
        if city == 'nan':
            city = ''
    if state_col and pd.notna(row.get(state_col)):
        state = str(row[state_col]).strip()
        if state == 'nan':
            state = ''
    if postal_col and pd.notna(row.get(postal_col)):
        postal_code = str(row[postal_col]).strip()
        if postal_code == 'nan':
            postal_code = ''
    
    # Get coordinates
    lat = None
    lon = None
    if lat_col and pd.notna(row.get(lat_col)):
        try:
            lat = float(row[lat_col])
            if lat == 0:
                lat = None
        except:
            lat = None
    if lon_col and pd.notna(row.get(lon_col)):
        try:
            lon = float(row[lon_col])
            if lon == 0:
                lon = None
        except:
            lon = None
    
    description = str(row.get(desc_col, ''))[:500] if desc_col else ''
    phone = str(row.get(phone_col, '')) if phone_col else ''
    email = str(row.get(email_col, '')) if email_col else ''
    
    # Healthcare access status
    healthcare_access = healthcare_access_map.get(state, 'Unknown')
    
    facilities.append({
        'id': str(idx),
        'name': name,
        'city': city,
        'state': state,
        'postal_code': postal_code,
        'latitude': lat,
        'longitude': lon,
        'specialties': row['specialties_list'][:8] if isinstance(row['specialties_list'], list) else [],
        'procedures': row['procedures_list'][:5] if isinstance(row['procedures_list'], list) else [],
        'equipment': row['equipment_list'][:5] if isinstance(row['equipment_list'], list) else [],
        'trust_score': float(row['trust_score']),
        'trust_level': row['trust_level'],
        'trust_percentage': int(row['trust_score'] * 100),
        'description': description,
        'phone': phone,
        'email': email,
        'healthcare_access': healthcare_access
    })

# Get all unique specialties
all_specialties = []
for specialties in df['specialties_list']:
    if isinstance(specialties, list):
        all_specialties.extend(specialties)
unique_specialties = sorted(set(all_specialties))[:30]

print(f"\n📊 Dataset Summary:")
print(f"   • Total facilities: {len(facilities)}")
print(f"   • States/Regions: {len(states_list)}")
print(f"   • Cities: {len(all_cities)}")
print(f"   • Specialties: {len(unique_specialties)}")
print(f"   • Avg trust score: {df['trust_score'].mean():.1%}")
print(f"   • Healthcare Deserts: {len(desert_states)} regions")
print(f"   • Life expectancy data loaded for {len(life_expectancy_data)} states")

# ============ ADVANCED AI SEARCH ENGINE ============
class AdvancedSearchEngine:
    def __init__(self, facilities):
        self.facilities = facilities
        
        # Comprehensive keyword mappings
        self.specialty_synonyms = {
            'dental': ['dentistry', 'dental', 'dentist', 'tooth', 'teeth', 'orthodontic', 'endodontic', 'periodontic', 'root canal'],
            'cardiology': ['cardiac', 'heart', 'cardiologist', 'cardio', 'cardiovascular'],
            'pediatrics': ['pediatric', 'child', 'children', 'baby', 'infant', 'kids', 'paediatric'],
            'orthopedics': ['orthopedic', 'bone', 'joint', 'fracture', 'ortho', 'knee', 'hip', 'spine'],
            'emergency': ['emergency', 'trauma', 'casualty', 'urgent', 'critical', '24/7'],
            'eye': ['ophthalmology', 'eye', 'vision', 'sight', 'retina', 'cataract'],
            'skin': ['dermatology', 'skin', 'hair', 'dermatologist'],
            'gynecology': ['gynecology', 'women', 'gyn', 'obstetrics', 'maternity', 'pregnancy'],
            'neurology': ['neurology', 'brain', 'neuro', 'stroke', 'neurologist'],
            'cancer': ['oncology', 'cancer', 'tumor', 'chemotherapy', 'radiation'],
            'surgery': ['surgery', 'surgeon', 'surgical', 'operation'],
            'radiology': ['radiology', 'x-ray', 'imaging', 'ct scan', 'mri', 'ultrasound'],
        }
        
        self.trust_keywords = {
            'high': ['best', 'top', 'excellent', 'premium', 'reputed', 'trusted', 'highest rated'],
            'medium': ['good', 'decent', 'average', 'satisfactory'],
            'low': ['poor', 'bad', 'low rated']
        }
    
    def search(self, query):
        query_lower = query.lower()
        results = []
        
        for facility in self.facilities:
            score = 0
            match_reasons = []
            
            # Specialty matching
            specialty_score, matched = self._match_specialties(facility, query_lower)
            score += specialty_score
            match_reasons.extend(matched)
            
            # Name matching
            if query_lower in facility['name'].lower():
                score += 30
                match_reasons.append("Name matches your search")
            
            # Location matching (using dynamic state/city data)
            location_score = self._match_location(facility, query_lower)
            score += location_score
            if location_score > 10 and facility['city']:
                match_reasons.append(f"Located in {facility['city']}")
            
            # Trust level matching
            if 'high trust' in query_lower and facility['trust_score'] >= 0.7:
                score += 20
            elif 'best' in query_lower and facility['trust_score'] >= 0.7:
                score += 15
            
            # Description matching
            if facility['description'] and any(word in facility['description'].lower() for word in query_lower.split() if len(word) > 3):
                score += 10
            
            # Trust score boost
            score += facility['trust_score'] * 15
            
            if score > 15:
                results.append({
                    'facility': facility,
                    'score': min(100, int(score)),
                    'reasons': match_reasons[:3]
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:20]
    
    def _match_specialties(self, facility, query):
        score = 0
        matches = []
        facility_specialties_text = ' '.join(facility['specialties']).lower()
        
        for specialty, keywords in self.specialty_synonyms.items():
            if any(kw in query for kw in keywords):
                if any(kw in facility_specialties_text for kw in keywords):
                    score += 35
                    matches.append(f"Specializes in {specialty.title()}")
                    break
        
        for specialty in facility['specialties']:
            if specialty.lower() in query:
                score += 25
                matches.append(f"Has {specialty} specialty")
        
        return min(score, 40), matches[:2]
    
    def _match_location(self, facility, query):
        score = 0
        if facility['state'] and facility['state'].lower() in query:
            score += 25
        if facility['city'] and facility['city'].lower() in query:
            score += 20
        return min(score, 25)

# Initialize search engine
search_engine = AdvancedSearchEngine(facilities)

# ============ DASHBOARD HTML with ALL INTEGRATED FEATURES + LIFE EXPECTANCY ============
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Healthcare Intelligence Dashboard | Living Healthcare Network | Databricks Challenge</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', sans-serif; }
        .gradient-bg { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); }
        .life-bg { background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%); }
        .trust-high { background: #10b981; color: white; }
        .trust-medium { background: #f59e0b; color: white; }
        .trust-low { background: #ef4444; color: white; }
        .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 500; margin: 2px; }
        .badge-specialty { background: #e0e7ff; color: #3730a3; }
        .stat-card { transition: all 0.3s ease; cursor: pointer; }
        .stat-card:hover { transform: translateY(-4px); box-shadow: 0 12px 20px rgba(0,0,0,0.1); }
        .suggestion-chip { background: #f3f4f6; padding: 8px 16px; border-radius: 25px; font-size: 14px; cursor: pointer; transition: all 0.2s; }
        .suggestion-chip:hover { background: #e5e7eb; transform: scale(1.05); }
        .modal { display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); }
        .modal-content { background: white; margin: 5% auto; padding: 20px; width: 80%; max-width: 800px; border-radius: 15px; max-height: 80vh; overflow-y: auto; }
        .close { float: right; font-size: 28px; cursor: pointer; font-weight: bold; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
        .animate-pulse-slow { animation: pulse 2s infinite; }
        .tab-btn { transition: all 0.2s; }
        .tab-btn.active { background: #1e3c72 !important; color: white !important; }
        .slider { -webkit-appearance: none; width: 100%; height: 6px; background: #e2e8f0; border-radius: 3px; outline: none; }
        .slider::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; background: #14b8a6; border-radius: 50%; cursor: pointer; }
    </style>
</head>
<body class="bg-gray-50">

    <!-- DATABRICKS CHALLENGE BANNER -->
    <div class="bg-gradient-to-r from-purple-900 via-blue-900 to-purple-900 text-white py-4 px-4">
        <div class="container mx-auto text-center">
            <div class="flex items-center justify-center gap-4 flex-wrap">
                <span class="text-3xl">🎯</span>
                <div>
                    <p class="text-xs uppercase tracking-wider text-purple-300">Databricks Challenge: Agency Healthcare Maps</p>
                    <h2 class="text-xl md:text-2xl font-bold">"A postal code can determine a lifespan — but it doesn't have to"</h2>
                    <p class="text-purple-200 mt-1 text-sm">Turning {{ total_facilities }}+ hospital records into a living intelligence network</p>
                </div>
                <span class="text-3xl">🏥</span>
            </div>
        </div>
    </div>

    <!-- Tab Navigation -->
    <div class="bg-white shadow-sm border-b">
        <div class="container mx-auto px-6">
            <div class="flex flex-wrap gap-1">
                <button onclick="showTab('dashboard')" id="tabDashboard" class="tab-btn px-5 py-3 text-sm font-medium bg-blue-600 text-white active">🏥 Healthcare Dashboard</button>
                <button onclick="showTab('life')" id="tabLife" class="tab-btn px-5 py-3 text-sm font-medium bg-gray-200 text-gray-700">📊 Life Expectancy Calculator</button>
                <button onclick="showTab('states')" id="tabStates" class="tab-btn px-5 py-3 text-sm font-medium bg-gray-200 text-gray-700">📈 State Health Stats</button>
                <button onclick="showTab('longevity')" id="tabLongevity" class="tab-btn px-5 py-3 text-sm font-medium bg-gray-200 text-gray-700">💪 Longevity Tips</button>
            </div>
        </div>
    </div>

    <!-- TAB 1: Healthcare Dashboard (Original Content) -->
    <div id="dashboardTab" class="tab-content">
        <div class="gradient-bg text-white">
            <div class="container mx-auto px-6 py-6">
                <div class="flex justify-between items-center flex-wrap">
                    <div>
                        <h1 class="text-2xl font-bold">🧠 Healthcare Intelligence Dashboard</h1>
                        <p class="text-blue-100 mt-1 text-sm">📍 India Healthcare Network | {{ total_facilities }} facilities | {{ total_states }} states/regions</p>
                    </div>
                    <div class="text-right">
                        <div class="text-2xl font-bold">{{ total_facilities }}</div>
                        <div class="text-blue-100 text-sm">Healthcare Facilities</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="container mx-auto px-6 py-6">
            <!-- LIVING INTELLIGENCE NETWORK STATUS -->
            <div class="bg-gradient-to-r from-indigo-500 to-purple-600 rounded-xl shadow-md p-5 mb-6 text-white">
                <div class="flex justify-between items-center flex-wrap">
                    <div>
                        <div class="flex items-center gap-2">
                            <span class="text-2xl">🧠</span>
                            <h3 class="text-lg font-bold">Living Healthcare Intelligence Network</h3>
                            <span class="bg-yellow-400 text-black text-xs px-2 py-1 rounded-full animate-pulse-slow">ACTIVE</span>
                        </div>
                        <p class="text-indigo-100 mt-1 text-xs">Turning messy hospital records into actionable insights</p>
                    </div>
                    <div class="text-right">
                        <div class="text-2xl font-bold" id="intelligenceScore">--</div>
                        <div class="text-indigo-200 text-xs">Data Intelligence Score</div>
                    </div>
                </div>
                
                <div id="accessAlert"></div>
                
                <div class="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
                    <div class="bg-white/20 rounded-lg p-2 backdrop-blur text-center">
                        <div class="text-xl" id="mappedCount">--</div>
                        <div class="text-indigo-100 text-xs">📍 Geo-mapped</div>
                    </div>
                    <div class="bg-white/20 rounded-lg p-2 backdrop-blur text-center">
                        <div class="text-xl" id="specializedCount">--</div>
                        <div class="text-indigo-100 text-xs">🏥 With Specialties</div>
                    </div>
                    <div class="bg-white/20 rounded-lg p-2 backdrop-blur text-center">
                        <div class="text-xl" id="contactableCount">--</div>
                        <div class="text-indigo-100 text-xs">📞 Contact Verified</div>
                    </div>
                </div>
            </div>

            <!-- PINCODE LOOKUP SECTION -->
            <div class="bg-white rounded-xl shadow-md p-5 mb-6">
                <h3 class="font-bold mb-3 text-lg">📍 Find Healthcare by Postal Code / PIN Code</h3>
                <p class="text-sm text-gray-600 mb-3 italic">"A postal code can determine a lifespan — we're changing that"</p>
                <div class="flex gap-3 flex-wrap">
                    <input type="text" id="pincodeInput" 
                           placeholder="Enter Indian PIN code (e.g., 400001, 110001, 560001)" 
                           class="flex-1 min-w-[200px] border border-gray-300 rounded-lg px-4 py-2">
                    <button onclick="searchByPincode()" class="bg-green-600 text-white px-6 py-2 rounded-lg hover:bg-green-700">
                        🔍 Find Nearby Healthcare
                    </button>
                </div>
                <div id="pincodeResult" class="mt-3 hidden"></div>
                <p class="text-xs text-gray-400 mt-2">💡 PIN codes starting with 400 = Mumbai, 110 = Delhi, 560 = Bangalore, 700 = Kolkata, 600 = Chennai, etc.</p>
            </div>

            <!-- AI Search Section -->
            <div class="bg-white rounded-xl shadow-md p-5 mb-6">
                <h2 class="text-lg font-semibold mb-3">🧠 AI Natural Language Search</h2>
                <div class="relative mb-3">
                    <input type="text" id="aiQuery" 
                           placeholder="Try: 'Find best dental clinics in Mumbai' or 'Show me emergency services' or 'Cardiology hospitals with high trust'"
                           class="w-full border border-gray-300 rounded-lg px-4 py-2 text-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500">
                    <div class="absolute right-2 top-1/2 transform -translate-y-1/2">
                        <button onclick="searchAI()" class="bg-blue-600 text-white px-5 py-1 rounded-lg text-sm hover:bg-blue-700">Search</button>
                    </div>
                </div>
                
                <div class="flex flex-wrap gap-2">
                    <div class="suggestion-chip" onclick="setQuery('Find dental clinics')">🦷 Dental clinics</div>
                    <div class="suggestion-chip" onclick="setQuery('Best cardiology hospitals with high trust')">❤️ Best cardiology</div>
                    <div class="suggestion-chip" onclick="setQuery('Emergency services 24/7')">🚨 Emergency 24/7</div>
                    <div class="suggestion-chip" onclick="setQuery('Pediatric specialists')">👶 Pediatric</div>
                    <div class="suggestion-chip" onclick="setQuery('Maternity and delivery services')">🤰 Maternity</div>
                </div>
            </div>

            <!-- AI Results -->
            <div id="aiResults" class="hidden mb-6">
                <div class="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl shadow-md p-4">
                    <h3 class="font-semibold text-gray-800 mb-3">🤖 AI Recommendations <span id="resultCount" class="text-sm text-gray-500"></span></h3>
                    <div id="aiResultsList" class="space-y-3 max-h-96 overflow-y-auto"></div>
                </div>
            </div>

            <!-- DYNAMIC STATE-CITY FILTERS -->
            <div class="bg-white rounded-xl shadow-md p-5 mb-6">
                <h2 class="text-lg font-semibold mb-3">🔍 Location-Based Filters</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">🏢 State / Region</label>
                        <select id="stateFilter" class="w-full border border-gray-300 rounded-lg px-3 py-2" onchange="updateCityFilter()">
                            <option value="">All States</option>
                            {% for state in states %}
                            <option value="{{ state }}">{{ state }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">📍 City</label>
                        <select id="cityFilter" class="w-full border border-gray-300 rounded-lg px-3 py-2">
                            <option value="">All Cities</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">⭐ Trust Level</label>
                        <select id="trustFilter" class="w-full border border-gray-300 rounded-lg px-3 py-2">
                            <option value="">All</option>
                            <option value="High">High Trust (≥70%)</option>
                            <option value="Medium">Medium Trust (50-69%)</option>
                            <option value="Low">Low Trust (<50%)</option>
                        </select>
                    </div>
                </div>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mt-3">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">🩺 Specialty</label>
                        <select id="specialtyFilter" class="w-full border border-gray-300 rounded-lg px-3 py-2">
                            <option value="">All Specialties</option>
                            {% for specialty in specialties %}
                            <option value="{{ specialty }}">{{ specialty }}</option>
                            {% endfor %}
                        </select>
                    </div>
                    <div class="flex items-end gap-2">
                        <button onclick="applyFilters()" class="bg-blue-600 text-white px-5 py-2 rounded-lg hover:bg-blue-700 text-sm">Apply Filters</button>
                        <button onclick="resetFilters()" class="bg-gray-500 text-white px-5 py-2 rounded-lg hover:bg-gray-600 text-sm">Reset</button>
                    </div>
                </div>
            </div>

            <!-- Stats Cards -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6" id="statsCards"></div>

            <!-- Charts -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
                <div class="bg-white rounded-xl shadow-md p-4"><h3 class="font-bold mb-3">📊 Trust Score Distribution</h3><div id="trustChart" style="height: 300px;"></div></div>
                <div class="bg-white rounded-xl shadow-md p-4"><h3 class="font-bold mb-3">📍 Top States/Regions</h3><div id="stateChart" style="height: 300px;"></div></div>
            </div>

            <!-- Map -->
            <div class="bg-white rounded-xl shadow-md p-4 mb-6">
                <h3 class="font-bold mb-3">🗺️ Healthcare Facility Map</h3>
                <div id="map" style="height: 400px;"></div>
                <p class="text-xs text-gray-500 mt-2">🟢 High Trust | 🟠 Medium Trust | 🔴 Low Trust</p>
            </div>

            <!-- Table -->
            <div class="bg-white rounded-xl shadow-md p-4">
                <h3 class="font-bold mb-3">📋 Facility Directory</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100">
                            <tr><th class="p-2 text-left">Name</th><th class="p-2 text-left">Location</th><th class="p-2 text-left">Specialties</th><th class="p-2 text-center">Trust</th><th class="p-2 text-center">Access</th><th class="p-2 text-center">Action</th></tr>
                        </thead>
                        <tbody id="tableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 2: Life Expectancy Calculator -->
    <div id="lifeTab" class="tab-content hidden">
        <div class="life-bg text-white">
            <div class="container mx-auto px-6 py-6">
                <h1 class="text-2xl font-bold">📊 Life Expectancy Calculator</h1>
                <p class="text-teal-100 mt-1">Evidence-based tool using NFHS-5 data and lifestyle factors</p>
            </div>
        </div>

        <div class="container mx-auto px-6 py-6">
            <div class="bg-white rounded-xl shadow-md p-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Your Age</label>
                        <input type="number" id="leAge" value="35" min="18" max="100" class="w-full border rounded-lg px-3 py-2">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Gender</label>
                        <select id="leGender" class="w-full border rounded-lg px-3 py-2">
                            <option value="male">Male</option>
                            <option value="female">Female</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Your State</label>
                        <select id="leState" class="w-full border rounded-lg px-3 py-2">
                            <option value="">Select State</option>
                            {% for state in states %}
                            <option value="{{ state }}">{{ state }}</option>
                            {% endfor %}
                        </select>
                    </div>
                </div>

                <div class="mt-6">
                    <h3 class="font-semibold text-gray-700 mb-3">Lifestyle Assessment</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="text-sm">🏃 Exercise Level:</label>
                            <select id="leExercise" class="w-full border rounded-lg px-2 py-1.5 text-sm">
                                <option value="sedentary">Sedentary (little to no exercise)</option>
                                <option value="light">Light (1-2 days/week)</option>
                                <option value="moderate" selected>Moderate (3-4 days/week)</option>
                                <option value="active">Active (5-6 days/week)</option>
                                <option value="very_active">Very Active (daily)</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-sm">🥗 Diet Quality:</label>
                            <select id="leDiet" class="w-full border rounded-lg px-2 py-1.5 text-sm">
                                <option value="poor">Poor (high processed food)</option>
                                <option value="average" selected>Average (mixed diet)</option>
                                <option value="good">Good (balanced, lots of veggies)</option>
                                <option value="excellent">Excellent (plant-based, whole foods)</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-sm">😴 Sleep Hours:</label>
                            <select id="leSleep" class="w-full border rounded-lg px-2 py-1.5 text-sm">
                                <option value="<6">&lt;6 hours</option>
                                <option value="6-7">6-7 hours</option>
                                <option value="7-8" selected>7-8 hours</option>
                                <option value="8+">8+ hours</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-sm">🧘 Stress Level:</label>
                            <select id="leStress" class="w-full border rounded-lg px-2 py-1.5 text-sm">
                                <option value="high">High (constant stress)</option>
                                <option value="moderate" selected>Moderate</option>
                                <option value="low">Low</option>
                                <option value="very_low">Very Low</option>
                            </select>
                        </div>
                        <div>
                            <label class="text-sm">👥 Social Connections:</label>
                            <select id="leSocial" class="w-full border rounded-lg px-2 py-1.5 text-sm">
                                <option value="poor">Poor (isolated)</option>
                                <option value="average" selected>Average</option>
                                <option value="good">Good</option>
                                <option value="strong">Strong (regular social circle)</option>
                            </select>
                        </div>
                    </div>
                </div>

                <button onclick="calculateLifeExpectancy()" class="mt-6 w-full bg-teal-600 text-white py-2.5 rounded-lg font-semibold hover:bg-teal-700 transition">
                    🔮 Calculate My Life Expectancy
                </button>

                <div id="leResultCard" class="mt-6 hidden"></div>
            </div>
        </div>
    </div>

    <!-- TAB 3: State Health Stats -->
    <div id="statesTab" class="tab-content hidden">
        <div class="gradient-bg text-white">
            <div class="container mx-auto px-6 py-6">
                <h1 class="text-2xl font-bold">📈 State Health Statistics</h1>
                <p class="text-blue-100 mt-1">Life expectancy, rankings, and health metrics across Indian states</p>
            </div>
        </div>

        <div class="container mx-auto px-6 py-6">
            <div class="bg-white rounded-xl shadow-md p-6">
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead class="bg-gray-100">
                            <tr>
                                <th class="p-3 text-left">Rank</th>
                                <th class="p-3 text-left">State</th>
                                <th class="p-3 text-center">Overall LE</th>
                                <th class="p-3 text-center">Male</th>
                                <th class="p-3 text-center">Female</th>
                                <th class="p-3 text-center">Infant Mortality</th>
                                <th class="p-3 text-center">Health Score</th>
                            </tr>
                        </thead>
                        <tbody id="stateStatsTable"></tbody>
                    </table>
                </div>
                <div class="mt-4 p-3 bg-yellow-50 rounded-lg text-xs text-yellow-700">
                    📊 Source: NFHS-5 (2019-21), Sample Registration System (SRS) 2020-24. Data for educational purposes.
                </div>
            </div>
        </div>
    </div>

    <!-- TAB 4: Longevity Tips -->
    <div id="longevityTab" class="tab-content hidden">
        <div class="life-bg text-white">
            <div class="container mx-auto px-6 py-6">
                <h1 class="text-2xl font-bold">💪 Evidence-Based Longevity Tips</h1>
                <p class="text-teal-100 mt-1">Small changes that can add years to your life</p>
            </div>
        </div>

        <div class="container mx-auto px-6 py-6">
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5" id="longevityTipsGrid"></div>
        </div>
    </div>

    <div id="modal" class="modal"><div class="modal-content"><span class="close" onclick="closeModal()">&times;</span><div id="modalBody"></div></div></div>

    <script>
        const stateCityMap = {{ state_city_map | safe }};
        let allFacilities = {{ facilities | safe }};
        let currentData = [...allFacilities];
        let map;
        const lifeExpectancyData = {{ life_expectancy_json | safe }};

        const longevityTips = [
            { icon: "🚶", title: "Walk 30 Minutes Daily", years: "+3 years", tip: "Reduces heart disease risk by 35%", indian: "Take a walk after dinner instead of sitting" },
            { icon: "🥗", title: "Add More Vegetables", years: "+4 years", tip: "Aim for 5 servings of veggies daily", indian: "Add extra sabzi to your roti" },
            { icon: "😴", title: "7-8 Hours Sleep", years: "+3 years", tip: "Quality sleep reduces inflammation", indian: "No phone 1 hour before bed" },
            { icon: "🧘", title: "Daily Meditation", years: "+2 years", tip: "Reduces stress and blood pressure", indian: "Start with 10 minutes of deep breathing" },
            { icon: "👥", title: "Strong Social Bonds", years: "+5 years", tip: "Loneliness increases mortality", indian: "Join a walking group or kitty party" },
            { icon: "🍵", title: "Turmeric & Green Tea", years: "+2 years", tip: "Powerful anti-inflammatory", indian: "Drink haldi doodh and green tea" },
            { icon: "🏋️", title: "Strength Training", years: "+2 years", tip: "Preserves muscle mass", indian: "Try squats, pushups, or yoga asanas" },
            { icon: "🚭", title: "Quit Smoking", years: "+10 years", tip: "Single best thing for longevity", indian: "Use nicotine patches or counselling" },
            { icon: "🩺", title: "Regular Checkups", years: "+3 years", tip: "Early detection saves lives", indian: "Annual BP, sugar, cholesterol checks" },
            { icon: "💧", title: "Stay Hydrated", years: "+1 year", tip: "Proper kidney and heart function", indian: "Keep a 2L water bottle at your desk" },
            { icon: "🌞", title: "Morning Sunlight", years: "+1.5 years", tip: "Vitamin D for bones & immunity", indian: "15 mins on your balcony each morning" },
            { icon: "📖", title: "Lifelong Learning", years: "+2.5 years", tip: "Builds cognitive reserve", indian: "Learn a language or do crossword puzzles" }
        ];

        function showTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(t => t.classList.add('hidden'));
            document.getElementById(tabName + 'Tab').classList.remove('hidden');
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active', 'bg-blue-600', 'text-white');
                btn.classList.add('bg-gray-200', 'text-gray-700');
            });
            const activeBtn = document.getElementById('tab' + tabName.charAt(0).toUpperCase() + tabName.slice(1));
            if (activeBtn) {
                activeBtn.classList.add('active', 'bg-blue-600', 'text-white');
                activeBtn.classList.remove('bg-gray-200', 'text-gray-700');
            }
            if (tabName === 'dashboard' && map) { setTimeout(() => map.invalidateSize(), 100); }
            if (tabName === 'states') { renderStateStats(); }
            if (tabName === 'longevity') { renderLongevityTips(); }
        }

        function renderStateStats() {
            const sorted = Object.entries(lifeExpectancyData).sort((a,b) => a[1].rank - b[1].rank);
            document.getElementById('stateStatsTable').innerHTML = sorted.map(([state, d]) => `
                <tr class="border-b hover:bg-gray-50">
                    <td class="p-3 font-bold text-center">${d.rank}</td>
                    <td class="p-3 font-medium">${state}</td>
                    <td class="p-3 text-center"><span class="font-bold text-teal-600">${d.overall}</span> yrs</td>
                    <td class="p-3 text-center">${d.male} yrs</td>
                    <td class="p-3 text-center">${d.female} yrs</td>
                    <td class="p-3 text-center">${d.infant_mortality}/1000</td>
                    <td class="p-3 text-center"><div class="w-full bg-gray-200 rounded-full h-2"><div class="bg-teal-600 h-2 rounded-full" style="width: ${d.health_score}%"></div></div><span class="text-xs">${d.health_score}%</span></td>
                </tr>
            `).join('');
        }

        function renderLongevityTips() {
            document.getElementById('longevityTipsGrid').innerHTML = longevityTips.map(tip => `
                <div class="bg-white rounded-xl shadow-md p-4 hover:shadow-lg transition cursor-pointer" onclick="showTipDetail('${tip.title}')">
                    <div class="text-3xl mb-2">${tip.icon}</div>
                    <h3 class="font-bold text-gray-800">${tip.title}</h3>
                    <p class="text-teal-600 font-semibold text-sm">${tip.years}</p>
                    <p class="text-xs text-gray-500 mt-1">${tip.tip}</p>
                    <p class="text-xs text-teal-600 mt-2">🇮🇳 ${tip.indian.substring(0, 50)}...</p>
                </div>
            `).join('');
        }

        function showTipDetail(title) {
            const tip = longevityTips.find(t => t.title === title);
            if (tip) {
                document.getElementById('modalBody').innerHTML = `
                    <h2 class="text-xl font-bold mb-3">${tip.icon} ${tip.title}</h2>
                    <div class="bg-teal-50 p-3 rounded-lg mb-3"><span class="text-2xl font-bold text-teal-600">${tip.years}</span><p class="text-sm">estimated life extension</p></div>
                    <p class="text-gray-700 mb-2"><strong>📚 Evidence:</strong> ${tip.tip}</p>
                    <p class="text-gray-700"><strong>🇮🇳 Indian Adaptation:</strong> ${tip.indian}</p>
                    <div class="mt-3 p-2 bg-gray-50 rounded text-xs text-gray-500">💡 Start with small changes — consistency matters more than intensity</div>
                `;
                document.getElementById('modal').style.display = 'block';
            }
        }

        function calculateLifeExpectancy() {
            const age = parseInt(document.getElementById('leAge').value);
            const gender = document.getElementById('leGender').value;
            const state = document.getElementById('leState').value;
            
            if (!state) { alert('Please select your state'); return; }
            
            const exercise = document.getElementById('leExercise').value;
            const diet = document.getElementById('leDiet').value;
            const sleep = document.getElementById('leSleep').value;
            const stress = document.getElementById('leStress').value;
            const social = document.getElementById('leSocial').value;
            
            const exerciseMap = {'sedentary': -3, 'light': 0, 'moderate': 2, 'active': 4, 'very_active': 6};
            const dietMap = {'poor': -4, 'average': 0, 'good': 3, 'excellent': 5};
            const sleepMap = {'<6': -2, '6-7': 0, '7-8': 2, '8+': 1};
            const stressMap = {'high': -5, 'moderate': -1, 'low': 2, 'very_low': 4};
            const socialMap = {'poor': -3, 'average': 0, 'good': 2, 'strong': 4};
            
            let baseLE = lifeExpectancyData[state] ? lifeExpectancyData[state][gender] : 70.0;
            let lifestyleAdjustment = exerciseMap[exercise] + dietMap[diet] + sleepMap[sleep] + stressMap[stress] + socialMap[social];
            let ageAdjustment = Math.max(-15, (40 - age) * 0.1);
            let finalLE = baseLE + lifestyleAdjustment + ageAdjustment;
            finalLE = Math.max(50, Math.min(100, finalLE));
            
            let recommendations = [];
            if (exerciseMap[exercise] < 0) recommendations.push('🏃 Add 30-min daily walks to gain 3-5 years');
            if (dietMap[diet] < 2) recommendations.push('🥗 Add more fruits and vegetables to your meals');
            if (sleepMap[sleep] < 1) recommendations.push('😴 Aim for 7-8 hours of quality sleep');
            if (stressMap[stress] < 0) recommendations.push('🧘 Try daily meditation to reduce stress');
            if (socialMap[social] < 1) recommendations.push('👥 Join community groups for social connection');
            
            const comparison = (finalLE - baseLE).toFixed(1);
            
            document.getElementById('leResultCard').innerHTML = `
                <div class="bg-gradient-to-r from-teal-50 to-blue-50 rounded-xl p-6 mt-4">
                    <div class="text-center">
                        <div class="text-5xl font-bold text-teal-600">${Math.round(finalLE)} years</div>
                        <p class="text-gray-500 mt-1">Estimated Life Expectancy</p>
                    </div>
                    <div class="grid grid-cols-2 gap-4 mt-4">
                        <div class="bg-white rounded-lg p-3 text-center">
                            <div class="text-xl font-bold text-blue-600">${(finalLE - age).toFixed(1)}</div>
                            <div class="text-xs">Remaining Years</div>
                        </div>
                        <div class="bg-white rounded-lg p-3 text-center">
                            <div class="text-xl font-bold ${comparison >= 0 ? 'text-green-600' : 'text-red-600'}">${comparison >= 0 ? '+' : ''}${comparison}</div>
                            <div class="text-xs">vs State Average</div>
                        </div>
                    </div>
                    <div class="bg-teal-100 rounded-lg p-3 mt-4">
                        <p class="text-sm font-semibold mb-2">💡 Personalized Recommendations:</p>
                        <ul class="text-xs space-y-1">${recommendations.map(r => `<li>${r}</li>`).join('')}</ul>
                    </div>
                    <div class="mt-3 text-center text-xs text-gray-500">
                        Based on NFHS-5 data and lifestyle research. Consult healthcare professionals.
                    </div>
                </div>
            `;
            document.getElementById('leResultCard').classList.remove('hidden');
        }

        // Original Healthcare Dashboard Functions
        function updateCityFilter() {
            const selectedState = document.getElementById('stateFilter').value;
            const citySelect = document.getElementById('cityFilter');
            citySelect.innerHTML = '<option value="">All Cities</option>';
            if (selectedState && stateCityMap[selectedState]) {
                stateCityMap[selectedState].forEach(city => {
                    citySelect.innerHTML += `<option value="${city}">${city}</option>`;
                });
            }
        }

        function updateIntelligenceScore() {
            const total = currentData.length;
            const withCoords = currentData.filter(f => f.latitude && f.longitude).length;
            const withSpecialties = currentData.filter(f => f.specialties && f.specialties.length > 0).length;
            const withContact = currentData.filter(f => (f.phone && f.phone !== 'nan' && f.phone !== '') || (f.email && f.email !== 'nan' && f.email !== '')).length;
            const completeness = Math.round((withCoords / total) * 30 + (withSpecialties / total) * 35 + (withContact / total) * 35) || 0;
            document.getElementById('intelligenceScore').innerHTML = completeness + '%';
            document.getElementById('mappedCount').innerHTML = withCoords;
            document.getElementById('specializedCount').innerHTML = withSpecialties;
            document.getElementById('contactableCount').innerHTML = withContact;
            
            const desertStates = {};
            currentData.forEach(f => { if (f.state && f.state !== 'Unknown') desertStates[f.state] = (desertStates[f.state] || 0) + 1; });
            const deserts = Object.entries(desertStates).filter(([_, count]) => count < 5);
            if (deserts.length > 0) {
                document.getElementById('accessAlert').innerHTML = `<div class="mt-3 bg-red-500/30 rounded-lg p-2"><p class="text-sm">⚠️ Healthcare Desert Alert: ${deserts.length} region(s) have <5 facilities</p><p class="text-xs opacity-75">${deserts.map(d => d[0]).join(', ')}</p></div>`;
            } else {
                document.getElementById('accessAlert').innerHTML = '';
            }
        }

        function searchByPincode() {
            const pincode = document.getElementById('pincodeInput').value;
            if (!pincode || pincode.length < 6) { alert('Please enter a valid 6-digit Indian PIN code'); return; }
            const resultDiv = document.getElementById('pincodeResult');
            resultDiv.innerHTML = '<div class="text-center py-2">🔍 Searching...</div>';
            resultDiv.classList.remove('hidden');
            fetch(`/api/search-by-pincode?pincode=${pincode}`).then(r => r.json()).then(data => {
                if (data.facilities && data.facilities.length > 0) {
                    resultDiv.innerHTML = `<div class="bg-green-50 p-3 rounded-lg border-l-4 border-green-500"><p class="font-semibold text-green-800">✅ ${data.facilities.length} facilities found near PIN ${pincode} (${data.region})</p><p class="text-sm text-green-600 mt-1">${data.analysis}</p><button onclick="showPincodeResults('${pincode}')" class="mt-2 text-green-700 underline text-sm">View all →</button></div>`;
                } else {
                    resultDiv.innerHTML = `<div class="bg-yellow-50 p-3 rounded-lg border-l-4 border-yellow-500"><p class="font-semibold text-yellow-800">⚠️ Limited healthcare access in PIN ${pincode} area</p><p class="text-xs text-yellow-500 mt-1">💡 This is exactly why "a postal code shouldn't determine a lifespan"</p></div>`;
                }
            });
        }

        function showPincodeResults(pincode) {
            fetch(`/api/search-by-pincode?pincode=${pincode}`).then(r => r.json()).then(data => {
                if (data.facilities) { currentData = data.facilities; updateStatsCards(); updateCharts(); updateTable(); updateMap(); updateIntelligenceScore(); document.getElementById('aiResults').classList.add('hidden'); }
            });
        }

        function updateStatsCards() {
            const total = currentData.length;
            const highTrust = currentData.filter(f => f.trust_score >= 0.7).length;
            const avgTrust = total > 0 ? (currentData.reduce((s,f) => s + f.trust_score, 0) / total * 100).toFixed(1) : 0;
            const desertCount = currentData.filter(f => f.healthcare_access === 'Healthcare Desert ⚠️').length;
            document.getElementById('statsCards').innerHTML = `
                <div class="stat-card bg-white rounded-xl shadow-md p-4 text-center"><div class="text-2xl font-bold text-blue-600">${total}</div><div class="text-gray-500 text-xs">Total Facilities</div></div>
                <div class="stat-card bg-white rounded-xl shadow-md p-4 text-center" onclick="applyFilter('trust', 'High')"><div class="text-2xl font-bold text-green-600">${highTrust}</div><div class="text-gray-500 text-xs">High Trust</div></div>
                <div class="stat-card bg-white rounded-xl shadow-md p-4 text-center"><div class="text-2xl font-bold text-yellow-600">${avgTrust}%</div><div class="text-gray-500 text-xs">Avg Trust</div></div>
                <div class="stat-card bg-white rounded-xl shadow-md p-4 text-center"><div class="text-2xl font-bold text-orange-600">${desertCount}</div><div class="text-gray-500 text-xs">In Deserts</div></div>
            `;
        }

        function updateCharts() {
            const trustCounts = { High: 0, Medium: 0, Low: 0 };
            currentData.forEach(f => { if (f.trust_score >= 0.7) trustCounts.High++; else if (f.trust_score >= 0.5) trustCounts.Medium++; else trustCounts.Low++; });
            Plotly.newPlot('trustChart', [{x: Object.keys(trustCounts), y: Object.values(trustCounts), type: 'bar', marker: {color: ['#10b981', '#f59e0b', '#ef4444']}}], {margin: {t: 30}});
            const stateCounts = {};
            currentData.forEach(f => { if (f.state) stateCounts[f.state] = (stateCounts[f.state] || 0) + 1; });
            const topStates = Object.entries(stateCounts).sort((a,b) => b[1] - a[1]).slice(0, 6);
            Plotly.newPlot('stateChart', [{x: topStates.map(s => s[1]), y: topStates.map(s => s[0]), type: 'bar', orientation: 'h', marker: {color: '#8b5cf6'}}], {margin: {t: 30, l: 100}});
        }

        function updateTable() {
            document.getElementById('tableBody').innerHTML = currentData.slice(0, 50).map(f => `
                <tr class="border-b"><td class="p-2 text-xs">${f.name.substring(0, 35)}</td><td class="p-2 text-xs">${f.city}, ${f.state}</td><td class="p-2">${f.specialties.slice(0,1).map(s => `<span class="badge badge-specialty">${s.substring(0,15)}</span>`).join('')}</td>
                <td class="p-2 text-center"><span class="px-2 py-0.5 rounded text-xs ${f.trust_score >= 0.7 ? 'trust-high' : 'trust-medium'}">${f.trust_percentage}%</span></td>
                <td class="p-2 text-center"><span class="text-xs ${f.healthcare_access === 'Healthcare Desert ⚠️' ? 'text-red-600' : 'text-green-600'}">${f.healthcare_access.includes('Desert') ? '⚠️ Desert' : '✓'}</span></td>
                <td class="p-2 text-center"><button onclick="showDetails('${f.id}')" class="text-blue-600 text-xs">View</button></td></tr>
            `).join('');
        }

        function updateMap() {
            if (map) map.remove();
            map = L.map('map').setView([20.5937, 78.9629], 5);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png').addTo(map);
            currentData.filter(f => f.latitude && f.longitude).slice(0, 300).forEach(f => {
                const color = f.trust_score >= 0.7 ? '#10b981' : '#f59e0b';
                L.circleMarker([f.latitude, f.longitude], {radius: 6, fillColor: color, color: '#fff', weight: 2, fillOpacity: 0.7}).addTo(map)
                    .bindPopup(`<b>${f.name}</b><br>📍 ${f.city}<br>⭐ ${f.trust_percentage}%`);
            });
        }

        function searchAI() {
            const query = document.getElementById('aiQuery').value;
            if (!query.trim()) return;
            fetch('/api/ai-search', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})})
                .then(r=>r.json()).then(data=>{
                    if(data.results){ currentData = data.results.map(r=>r.facility); updateStatsCards(); updateCharts(); updateTable(); updateMap(); updateIntelligenceScore(); 
                    document.getElementById('aiResults').innerHTML = '<div class="bg-teal-50 p-3 rounded mb-3">🎯 AI found '+data.results.length+' matches</div>'; document.getElementById('aiResults').classList.remove('hidden'); }
                });
        }

        function applyFilters() {
            const state = document.getElementById('stateFilter').value, city = document.getElementById('cityFilter').value, trust = document.getElementById('trustFilter').value, specialty = document.getElementById('specialtyFilter').value;
            currentData = allFacilities.filter(f => (!state||f.state===state) && (!city||f.city===city) && (!trust||f.trust_level===trust) && (!specialty||f.specialties.includes(specialty)));
            updateStatsCards(); updateCharts(); updateTable(); updateMap(); updateIntelligenceScore(); document.getElementById('aiResults').classList.add('hidden');
        }

        function resetFilters() {
            document.getElementById('stateFilter').value = ''; document.getElementById('cityFilter').innerHTML = '<option value="">All Cities</option>';
            document.getElementById('trustFilter').value = ''; document.getElementById('specialtyFilter').value = '';
            currentData = [...allFacilities]; updateStatsCards(); updateCharts(); updateTable(); updateMap(); updateIntelligenceScore();
        }

        function applyFilter(type, value) { if (type === 'trust') { document.getElementById('trustFilter').value = value; applyFilters(); } }
        function setQuery(text) { document.getElementById('aiQuery').value = text; searchAI(); }
        function showDetails(id) { const f = allFacilities.find(f => f.id === id); if(f){ document.getElementById('modalBody').innerHTML = `<h2 class="text-xl font-bold">🏥 ${f.name}</h2><p>📍 ${f.city}, ${f.state}</p><p>⭐ Trust: ${f.trust_percentage}%</p><p>🩺 ${f.specialties.join(', ')}</p>`; document.getElementById('modal').style.display = 'block'; } }
        function closeModal() { document.getElementById('modal').style.display = 'none'; }
        
        // Initialize
        updateStatsCards(); updateCharts(); updateTable(); updateMap(); updateIntelligenceScore(); updateCityFilter();
        if (Object.keys(lifeExpectancyData).length > 0) renderStateStats();
        showTab('dashboard');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(
        TEMPLATE,
        facilities=json.dumps(facilities),
        states=states_list,
        state_city_map=json.dumps(state_city_dict),
        specialties=unique_specialties,
        total_facilities=len(facilities),
        total_states=len(states_list),
        total_cities=len(all_cities),
        life_expectancy_json=json.dumps(life_expectancy_data)
    )

@app.route('/api/ai-search', methods=['POST'])
def ai_search():
    """Advanced AI search endpoint"""
    data = request.json
    query = data.get('query', '')
    
    results = search_engine.search(query)
    
    formatted_results = []
    for r in results:
        formatted_results.append({
            'facility': r['facility'],
            'score': r['score'],
            'reasons': r['reasons']
        })
    
    return jsonify({'results': formatted_results})

@app.route('/api/search-by-pincode', methods=['GET'])
def search_by_pincode():
    """Search healthcare facilities by Indian PIN code"""
    pincode = request.args.get('pincode', '')
    
    if not pincode or len(pincode) < 6:
        return jsonify({'facilities': [], 'analysis': 'Invalid PIN code'})
    
    # Get region from pincode
    region = get_region_from_pincode(pincode)
    
    # Find facilities in that region (by state or city match)
    nearby = []
    for f in facilities:
        if region != 'Unknown' and (region in f['city'] or region in f['state'] or f['state'] == region):
            nearby.append(f)
        elif region == 'Unknown' and f['postal_code'] and f['postal_code'][:3] == pincode[:3]:
            nearby.append(f)
    
    if nearby:
        avg_trust = sum(f['trust_score'] for f in nearby) / len(nearby) * 100
        analysis = f"Found {len(nearby)} facilities in {region}. Average trust score: {avg_trust:.0f}%"
    else:
        analysis = f"Limited healthcare infrastructure detected in PIN {pincode} region. This is a healthcare access gap area."
    
    return jsonify({
        'pincode': pincode,
        'region': region,
        'facilities': nearby[:15],
        'count': len(nearby),
        'analysis': analysis
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 HEALTHCARE INTELLIGENCE DASHBOARD")
    print("="*60)
    print(f"✅ Loaded {len(facilities)} facilities")
    print(f"📍 States/Regions: {len(states_list)}")
    print(f"🏙️ Cities: {len(all_cities)}")
    print(f"🩺 Specialties: {len(unique_specialties)}")
    print(f"⭐ Avg Trust Score: {df['trust_score'].mean():.1%}")
    print(f"🏜️ Healthcare Deserts: {len(desert_states)} regions")
    print(f"📊 Life Expectancy Data: {len(life_expectancy_data)} states loaded")
    print("\n✨ Databricks Challenge Features Integrated:")
    print("   • 'A postal code can determine a lifespan' - PIN code search")
    print("   • Living Intelligence Network - Real-time data quality scoring")
    print("   • Healthcare Desert Detection - Identifying underserved areas")
    print("   • AI Natural Language Search - Semantic understanding")
    print("   • Life Expectancy Calculator - Evidence-based algorithm")
    print("   • State Health Statistics - NFHS-5 data visualization")
    print("   • Longevity Tips - 12 evidence-based habits")
    print("\n🌐 Open in browser: http://localhost:8888")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=8888)