"""
File handling utilities for Excel operations
"""
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import io
import config

class FileHandler:
    """Handles Excel file operations for the preset comparison tool"""
    
    @staticmethod
    @st.cache_data(ttl=config.CACHE_TTL)
    def load_preset_database() -> pd.DataFrame:
        """
        Load the preset database from Excel file with caching
        
        Returns:
            pd.DataFrame: The preset database
            
        Raises:
            FileNotFoundError: If preset database file doesn't exist
            Exception: If file cannot be read
        """
        try:
            if not config.PRESET_DB_PATH.exists():
                raise FileNotFoundError(f"Preset database not found at {config.PRESET_DB_PATH}")
            
            # Read Excel file in chunks for large files
            df = pd.read_excel(config.PRESET_DB_PATH)
            
            # Validate required columns exist
            expected_cols = list(config.PRESET_COLUMNS.values())
            missing_cols = [col for col in expected_cols if col not in df.columns]
            
            if missing_cols:
                st.warning(f"Missing columns in preset database: {missing_cols}")
                # Try to map common variations
                df = FileHandler._map_column_variations(df, expected_cols)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading preset database: {str(e)}")
            raise
    
    @staticmethod
    def _map_column_variations(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        """Map common column name variations to expected names"""
        column_mappings = {
            # Preset database variations
            'category': ['Category', 'Cat', 'Type'],
            'sub_category': ['Sub-Category', 'Sub Category', 'SubCategory', 'Sub_Category'],
            'attribute_name': ['Attribute Name', 'Attribute', 'Attr Name', 'Parameter'],
            'preset_values': ['Preset Values', 'Preset values', 'Values', 'Preset Value', 'Reference Values'],
            # Input file variations  
            'category': ['Category', 'Main Category', 'Cat'],
            'sub_category': ['Sub-Category', 'Sub Category', 'SubCategory', 'Sub_Category'],
            'attribute_name': ['Attribute Name', 'Attribute', 'Attr Name', 'Parameter'],
            'input_values': ['Input values', 'Input Values', 'Value', 'Input Value', 'User Values']
        }
        
        for expected_col in expected_cols:
            if expected_col not in df.columns:
                # Find matching variation
                for variation in column_mappings.get(expected_col.lower().replace(' ', '_'), []):
                    if variation in df.columns:
                        df = df.rename(columns={variation: expected_col})
                        break
        
        return df
    
    @staticmethod
    def validate_input_file(uploaded_file) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Validate uploaded input file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple[bool, str, Optional[pd.DataFrame]]: (is_valid, message, dataframe)
        """
        try:
            # Check file size
            if uploaded_file.size > config.MAX_FILE_SIZE_MB * 1024 * 1024:
                return False, f"File size exceeds {config.MAX_FILE_SIZE_MB}MB limit", None
            
            # Check file extension
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in [f'.{ext}' for ext in config.ALLOWED_EXTENSIONS]:
                return False, f"File type {file_ext} not supported. Use: {config.ALLOWED_EXTENSIONS}", None
            
            # Read Excel file
            df = pd.read_excel(uploaded_file)
            
            if df.empty:
                return False, "File is empty", None
            
            # Map column variations
            expected_cols = list(config.INPUT_COLUMNS.values())
            df = FileHandler._map_column_variations(df, expected_cols)
            
            # Check required columns
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                available_cols = list(df.columns)
                return False, f"Missing required columns: {missing_cols}. Available columns: {available_cols}", None
            
            return True, f"File validated successfully. Found {len(df)} rows.", df
            
        except Exception as e:
            return False, f"Error reading file: {str(e)}", None
    
    @staticmethod
    def create_input_template() -> bytes:
        """
        Create a downloadable input template Excel file
        
        Returns:
            bytes: Excel file content as bytes
        """
        # Create sample data
        sample_data = {
            config.INPUT_COLUMNS['category']: [
                'Switches',
                'Switches', 
                'Connectors',
                'Resistors'
            ],
            config.INPUT_COLUMNS['sub_category']: [
                'Configurable Switch Components - Lens',
                'Configurable Switch Components - Illumination',
                'Terminal Blocks - Headers',
                'Fixed Resistors - Through Hole'
            ],
            config.INPUT_COLUMNS['attribute_name']: [
                'Compatible Series',
                'Compatible Series',
                'Contact Material',
                'Resistance'
            ],
            config.INPUT_COLUMNS['input_values']: [
                'EAO, 02',
                'IDEC, AL6',
                'Bronze',
                '10K Ohm'
            ]
        }
        
        df_template = pd.DataFrame(sample_data)
        
        # Convert to Excel bytes
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_template.to_excel(writer, index=False, sheet_name='Input Template')
            
            # Add instructions sheet
            instructions = pd.DataFrame({
                'Instructions': [
                    '1. Replace the sample data with your actual input values',
                    '2. Keep the column names EXACTLY as shown - they match your preset database',
                    '3. Category: Main product category (must match preset database categories)',
                    '4. Sub-Category: Detailed subcategory (must match preset database subcategories)',
                    '5. Attribute Name: Specific attribute to compare (e.g., "Compatible Series", "Resistance")',
                    '6. Input values: Your input values to compare against preset values',
                    '7. Upload this file to the comparison tool',
                    '',
                    'How the comparison works:',
                    '- The tool matches your "Attribute Name" with preset database attributes',
                    '- Then compares your "Input values" against corresponding "Preset values"',
                    '- Results show similarity scores and match explanations',
                    '',
                    'Supported value formats:',
                    '- Plain text: "EAO, 02"',
                    '- Numbers with units: "20 kg", "0.181 inches"', 
                    '- Multiple values: "20 kg, 15 cm"',
                    '- Values with conditions: "20 kg @ 30°C"',
                    '- Complex specifications: "Bronze, Nickel Plated"'
                ]
            })
            instructions.to_excel(writer, index=False, sheet_name='Instructions')
        
        output.seek(0)
        return output.read()
    
    @staticmethod
    def backup_preset_database() -> bool:
        """
        Create a backup of the current preset database
        
        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            backup_path = config.DATA_DIR / f"preset_database_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            if config.PRESET_DB_PATH.exists():
                df = pd.read_excel(config.PRESET_DB_PATH)
                df.to_excel(backup_path, index=False)
                return True
            return False
            
        except Exception as e:
            st.error(f"Error creating backup: {str(e)}")
            return False
    
    @staticmethod
    def update_preset_database(new_file) -> Tuple[bool, str]:
        """
        Update the preset database with a new file
        
        Args:
            new_file: Streamlit uploaded file object
            
        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            # Validate new file structure
            df = pd.read_excel(new_file)
            expected_cols = list(config.PRESET_COLUMNS.values())
            
            # Map column variations
            df = FileHandler._map_column_variations(df, expected_cols)
            
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                return False, f"New database missing required columns: {missing_cols}"
            
            # Create backup
            backup_success = FileHandler.backup_preset_database()
            if not backup_success:
                st.warning("Could not create backup, but proceeding with update")
            
            # Update database
            df.to_excel(config.PRESET_DB_PATH, index=False)
            
            # Clear cache to reload new data
            st.cache_data.clear()
            
            return True, f"Database updated successfully with {len(df)} records"
            
        except Exception as e:
            return False, f"Error updating database: {str(e)}"