import pandas as pd
import numpy as np

class ClinicalTextBridge:
    def __init__(self):
        pass

    def _get_value(self, row, keys, default="Unknown"):
        for k in keys:
            if k in row and pd.notna(row[k]):
                return row[k]
        return default

    def generate_text(self, row, source_type='icbhi'):
        # 1. Demographics
        age = self._get_value(row, ['age', 'Age_P1', 'Age'], default=None)
        age_str = f"{int(age)}-year-old" if age else "Adult"
        
        # FIX: Force conversion to string before .lower()
        # This prevents AttributeError when sex is stored as int (1 or 2)
        sex_raw = self._get_value(row, ['sex', 'gender', 'Sex'], default='')
        sex_val = str(sex_raw).lower()
        
        # Logic for parsing sex (handling both 'F'/'M' and 1/2 integer codes)
        # Common medical coding: 1=Male, 2=Female
        if 'f' in sex_val or '2' in sex_val: sex_str = "female" 
        elif 'm' in sex_val or '1' in sex_val: sex_str = "male"
        else: sex_str = "patient"

        disease_str = "respiratory condition"
        sound = ""

        # 2. Logic for ICBHI (Audio Source)
        if source_type == 'icbhi':
            d = row.get('disease', 'Unknown')
            if d == 'COPD': disease_str = "Chronic Obstructive Pulmonary Disease"
            elif d == 'Healthy': disease_str = "Healthy condition"
            else: disease_str = d
            
            c = row.get('crackles', 0)
            w = row.get('wheezes', 0)
            if c==1 and w==1: sound = " Auscultation reveals crackles and wheezes."
            elif c==1: sound = " Auscultation reveals crackles."
            elif w==1: sound = " Auscultation reveals wheezes."
            else: sound = " Auscultation reveals normal breathing."
            
            return f"A {age_str} {sex_str} diagnosed with {disease_str}.{sound}"

        # 3. Logic for COPDGene (Omics Source)
        elif source_type == 'omics':
            gold = self._get_value(row, ['finalGold_P1', 'final_Gold_Stage'])
            try:
                gold = int(float(gold))
                if gold == 0: disease_str = "Healthy condition"
                elif gold >= 3: disease_str = "Severe Chronic Obstructive Pulmonary Disease"
                else: disease_str = "COPD"
            except:
                pass
            return f"A {age_str} {sex_str} presenting with {disease_str}. Clinical proteomics profile."

        return "Patient data."