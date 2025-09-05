"""
Main Streamlit application for the Preset Comparison Tool
"""
import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import project modules
import config
from utils.file_handler import FileHandler
from utils.data_processor import DataProcessor
from utils.matching_engine import EnhancedMatchingEngine, MatchResult
from utils.export_manager import ExportManager, ResultsAnalyzer
from components.ui_components import UIComponents
from components.sidebar import Sidebar, FilterPanel
from components.results_display import ResultsDisplay

class PresetComparisonApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize the application"""
        self.initialize_session_state()
        UIComponents.render_header()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'results_data' not in st.session_state:
            st.session_state.results_data = []
        
        if 'input_data' not in st.session_state:
            st.session_state.input_data = None
        
        if 'preset_data' not in st.session_state:
            st.session_state.preset_data = None
        
        if 'processing_complete' not in st.session_state:
            st.session_state.processing_complete = False
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'compare'
    
    def run(self):
        """Run the main application"""
        # Render sidebar and get state
        sidebar_state = Sidebar.render()
        
        # Handle database actions
        self.handle_database_actions(sidebar_state.get('database_action'))
        
        # Route to appropriate page
        page = sidebar_state.get('page', 'compare')
        
        if page == 'home':
            self.render_home_page()
        elif page == 'compare':
            self.render_comparison_page(sidebar_state.get('processing_options', {}))
        elif page == 'analysis':
            self.render_analysis_page()
        elif page == 'settings':
            self.render_settings_page()
        elif page == 'help':
            self.render_help_page()
    
    def handle_database_actions(self, action: Optional[str]):
        """Handle database management actions"""
        if not action:
            return
        
        if action == "refresh":
            st.cache_data.clear()
            st.success("Database cache refreshed!")
        
        elif action == "backup":
            success = FileHandler.backup_preset_database()
            if success:
                st.success("Database backup created successfully!")
            else:
                st.error("Failed to create database backup")
        
        elif action == "update":
            if 'new_db_file' in st.session_state:
                success, message = FileHandler.update_preset_database(st.session_state.new_db_file)
                if success:
                    st.success(message)
                    st.session_state.preset_data = None  # Force reload
                else:
                    st.error(message)
                del st.session_state.new_db_file
    
    def render_home_page(self):
        """Render home/welcome page"""
        st.header("🏠 Welcome to the Preset Comparison Tool")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### What This Tool Does
            
            This intelligent comparison tool helps you match your input values against a comprehensive preset database using:
            
            - **🎯 Exact Matching**: Find perfect matches instantly
            - **🔍 Fuzzy Matching**: Discover similar values with advanced algorithms
            - **🧠 Smart Analysis**: Handle complex formats, units, and conditions
            - **📊 Detailed Reporting**: Get insights and export results
            
            ### Getting Started
            
            1. **📥 Download Template**: Get the Excel template for your input data
            2. **✏️ Fill Your Data**: Add your values using the template format
            3. **📤 Upload & Compare**: Upload your file and let the tool do the work
            4. **📋 Review Results**: Examine matches and export your results
            
            ### Supported Value Types
            
            - Plain text values
            - Numbers with units (kg, inches, °C, etc.)
            - Multiple values (comma-separated)
            - Values with conditions (@ symbol)
            - Complex dimensions and specifications
            """)
        
        with col2:
            st.subheader("📊 Database Status")
            try:
                preset_data = self.load_preset_database()
                if preset_data is not None:
                    st.success(f"✅ Database loaded")
                    st.info(f"**Records**: {len(preset_data):,}")
                    st.info(f"**Categories**: {preset_data.iloc[:, 0].nunique() if len(preset_data.columns) > 0 else 0}")
                else:
                    st.error("❌ Database not available")
            except Exception as e:
                st.error(f"❌ Database error: {str(e)}")
            
            # Quick action buttons
            st.subheader("🚀 Quick Actions")
            
            if st.button("📋 Download Template", use_container_width=True):
                st.session_state.current_page = 'compare'
                st.rerun()
            
            if st.button("🔍 Start Comparison", use_container_width=True):
                st.session_state.current_page = 'compare'  
                st.rerun()
    
    def render_comparison_page(self, processing_options: Dict[str, Any]):
        """Render the main comparison page"""
        st.header("🔍 Compare Your Values")
        
        # Load preset database
        preset_data = self.load_preset_database()
        if preset_data is None:
            st.error("❌ Could not load preset database. Please check the database file.")
            return
        
        # File upload and template download section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = UIComponents.render_file_uploader()
        
        with col2:
            UIComponents.render_template_download()
        
        # Process uploaded file
        if uploaded_file is not None:
            self.process_uploaded_file(uploaded_file, preset_data, processing_options)
        
        # Show results if available
        if st.session_state.results_data:
            st.divider()
            self.render_results_section()
    
    def process_uploaded_file(self, uploaded_file, preset_data: pd.DataFrame, 
                            processing_options: Dict[str, Any]):
        """Process the uploaded file and perform comparison"""
        
        # Validate uploaded file
        is_valid, message, input_df = FileHandler.validate_input_file(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {message}")
            return
        
        st.success(f"✅ {message}")
        
        # Show preview of input data
        with st.expander("👀 Preview Input Data", expanded=False):
            st.dataframe(input_df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(input_df)} rows")
        
        # Process button
        if st.button("🚀 Start Comparison", type="primary", use_container_width=True):
            self.run_comparison(input_df, preset_data, processing_options)
    
    def run_comparison(self, input_df: pd.DataFrame, preset_data: pd.DataFrame,
                      processing_options: Dict[str, Any]):
        """Run the comparison process"""
        
        start_time = time.time()
        
        # Create progress indicators
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        try:
            # Initialize matching engine
            status_text.text("🔧 Initializing matching engine...")
            progress_bar.progress(0.1)
            
            # Apply processing options
            config.MATCHING_THRESHOLD = processing_options.get('threshold', config.MATCHING_THRESHOLD)
            config.MAX_RESULTS_PER_INPUT = processing_options.get('max_results', config.MAX_RESULTS_PER_INPUT)
            
            matching_engine = EnhancedMatchingEngine(preset_data)
            
            # Get input values column
            value_columns = [col for col in input_df.columns if 'input values' in col.lower() or 'value' in col.lower()]
            if not value_columns:
                st.error("Could not find input values column in input data. Expected 'Input values' column.")
                return
            
            input_column = value_columns[0]
            
            status_text.text(f"🔍 Processing {len(input_df)} rows with composite key matching...")
            progress_bar.progress(0.2)
            
            # Process each input row with composite key matching
            all_results = []
            total_rows = len(input_df)
            
            for i, (idx, row) in enumerate(input_df.iterrows()):
                # Update progress
                progress = 0.2 + (0.7 * (i + 1) / total_rows)
                progress_bar.progress(progress)
                
                # Extract context information
                category = str(row.get('Category', '')) if pd.notna(row.get('Category')) else None
                sub_category = str(row.get('Sub-Category', '')) if pd.notna(row.get('Sub-Category')) else None
                attribute_name = str(row.get('Attribute Name', '')) if pd.notna(row.get('Attribute Name')) else None
                input_value = str(row.get(input_column, '')) if pd.notna(row.get(input_column)) else None
                
                if not input_value or not category or not attribute_name:
                    # Skip rows with missing essential information
                    continue
                
                # Create composite key for context
                composite_key = f"{category} > {sub_category} > {attribute_name}" if sub_category else f"{category} > {attribute_name}"
                
                status_text.text(f"🔍 Processing: {composite_key[:50]}...")
                
                # Find matches using composite key
                matches = matching_engine.find_matches(
                    input_value,
                    category=category,
                    sub_category=sub_category, 
                    attribute_name=attribute_name
                )
                match_result = MatchResult(input_value, matches)
                
                # Convert to result rows with context information
                result_rows = match_result.to_result_rows(category, sub_category, attribute_name)
                
                all_results.extend(result_rows)
                
                # Add a small delay to show progress
                time.sleep(0.01)
            
            # Finalize results
            status_text.text("📊 Finalizing results...")
            progress_bar.progress(0.9)
            
            # Store results in session state
            st.session_state.results_data = all_results
            st.session_state.input_data = input_df
            st.session_state.preset_data = preset_data
            st.session_state.processing_complete = True
            
            # Store processing stats
            processing_time = time.time() - start_time
            st.session_state.processing_stats = {
                'duration': f"{processing_time:.2f}s",
                'records': len(all_results),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            st.session_state.last_processing_time = st.session_state.processing_stats['timestamp']
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            time.sleep(1)  # Brief pause to show completion
            progress_container.empty()  # Remove progress indicators
            
            # Show success message
            st.success(f"🎉 Comparison complete! Processed {len(input_df)} input rows with composite key matching in {processing_time:.2f}s")
            
            # Auto-scroll to results
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)
    
    def render_results_section(self):
        """Render the results section"""
        st.header("📊 Comparison Results")
        
        # Results overview
        ResultsDisplay.render_results_overview(st.session_state.results_data)
        
        st.divider()
        
        # Filter panel
        st.subheader("🔍 Filter & Search")
        filters = FilterPanel.render(st.session_state.results_data)
        
        # Detailed results
        filtered_df = ResultsDisplay.render_detailed_results(
            st.session_state.results_data, 
            filters
        )
        
        st.divider()
        
        # Export section
        ResultsDisplay.render_export_section(st.session_state.results_data)
        
        # Insights section
        ResultsDisplay.render_insights_section(st.session_state.results_data)
    
    def render_analysis_page(self):
        """Render analysis page for detailed insights"""
        st.header("📊 Results Analysis")
        
        if not st.session_state.results_data:
            st.info("No results available for analysis. Please run a comparison first.")
            
            if st.button("🔍 Go to Comparison", use_container_width=True):
                st.session_state.current_page = 'compare'
                st.rerun()
            return
        
        # Deep analysis
        analysis = ResultsAnalyzer.analyze_results(st.session_state.results_data)
        
        # Summary metrics
        ResultsDisplay.render_results_overview(st.session_state.results_data)
        
        st.divider()
        
        # Advanced charts and insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Advanced Analytics")
            
            # Performance metrics
            if 'processing_stats' in st.session_state:
                stats = st.session_state.processing_stats
                st.metric("Processing Time", stats['duration'])
                st.metric("Records/Second", f"{int(stats['records']) / float(stats['duration'].replace('s', '')):.1f}")
            
            # Quality metrics
            df = pd.DataFrame(st.session_state.results_data)
            exact_rate = len(df[df['Status'] == config.MatchStatus.EXACT_MATCH]) / len(df) * 100
            
            st.metric("Data Quality Score", f"{exact_rate:.1f}%")
        
        with col2:
            st.subheader("💡 Recommendations")
            
            if 'insights' in analysis:
                for insight in analysis['insights']:
                    st.info(f"💡 {insight}")
            else:
                st.info("No specific recommendations at this time.")
        
        # Detailed breakdown
        st.subheader("🔍 Detailed Breakdown")
        
        # Filter panel for analysis
        filters = FilterPanel.render(st.session_state.results_data)
        
        # Show filtered results
        if filters:
            filtered_df = FilterPanel.apply_filters(pd.DataFrame(st.session_state.results_data), filters)
            st.write(f"**Filtered Results:** {len(filtered_df)} items")
            
            if not filtered_df.empty:
                ResultsDisplay._render_distribution_charts(filtered_df)
    
    def render_settings_page(self):
        """Render settings page"""
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Matching Settings")
            
            # Current settings display
            st.write("**Current Configuration:**")
            st.code(f"""
Similarity Threshold: {config.MATCHING_THRESHOLD * 100}%
Exact Match Threshold: {config.EXACT_MATCH_THRESHOLD * 100}%
Max Results per Input: {config.MAX_RESULTS_PER_INPUT}
Max File Size: {config.MAX_FILE_SIZE_MB}MB
""")
            
            # Performance settings
            st.subheader("⚡ Performance")
            st.write(f"**Cache TTL:** {config.CACHE_TTL} seconds")
            st.write(f"**Chunk Size:** {config.CHUNK_SIZE} records")
            st.write(f"**Memory Limit:** {config.MAX_MEMORY_USAGE_MB}MB")
        
        with col2:
            st.subheader("🗄️ Database Information")
            
            try:
                preset_data = self.load_preset_database()
                if preset_data is not None:
                    st.success("✅ Database Connected")
                    
                    # Database stats
                    st.write("**Database Statistics:**")
                    st.write(f"• Total Records: {len(preset_data):,}")
                    st.write(f"• Columns: {len(preset_data.columns)}")
                    st.write(f"• Memory Usage: {preset_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                    
                    # Column information
                    st.write("**Columns:**")
                    for col in preset_data.columns:
                        non_null = preset_data[col].notna().sum()
                        st.write(f"• {col}: {non_null:,} values")
                
                else:
                    st.error("❌ Database Not Available")
                    
            except Exception as e:
                st.error(f"❌ Database Error: {str(e)}")
            
            # Database actions
            st.subheader("🔧 Database Management")
            
            if st.button("🔄 Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared successfully!")
            
            if st.button("💾 Create Backup", use_container_width=True):
                success = FileHandler.backup_preset_database()
                if success:
                    st.success("Backup created successfully!")
                else:
                    st.error("Failed to create backup")
        
        # Reset session state
        st.divider()
        st.subheader("🔄 Reset Application")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Results", use_container_width=True):
                st.session_state.results_data = []
                st.session_state.processing_complete = False
                st.success("Results cleared!")
        
        with col2:
            if st.button("🔄 Reset All", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key.startswith(('results_', 'input_', 'preset_', 'processing_')):
                        del st.session_state[key]
                st.success("Application reset!")
    
    def render_help_page(self):
        """Render help and documentation page"""
        st.header("❓ Help & Documentation")
        
        # Render help section
        UIComponents.render_help_section()
        
        # Additional help sections
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 Troubleshooting")
            
            with st.expander("Common Issues"):
                st.markdown("""
                **File Upload Issues:**
                - Ensure file is in Excel format (.xlsx)
                - Check file size is under 50MB
                - Verify all required columns are present
                
                **No Matches Found:**
                - Check input value formatting
                - Lower the similarity threshold
                - Review spelling and terminology
                
                **Slow Performance:**
                - Reduce input file size
                - Clear cache in settings
                - Use fewer results per input
                """)
        
        with col2:
            st.subheader("📞 Support")
            
            st.info("""
            **Version:** """ + config.VERSION + """
            
            **Features:**
            - Advanced fuzzy matching
            - Multi-format value support
            - Interactive results analysis
            - Excel/CSV export
            - Real-time processing
            """)
            
            st.write("**Sample Data:**")
            sample_df = pd.DataFrame({
                'Original Input': config.SAMPLE_INPUTS[:3],
                'Expected': ['Should find matches', 'Unit conversion', 'Exact match']
            })
            st.dataframe(sample_df, use_container_width=True)
    
    def load_preset_database(self) -> Optional[pd.DataFrame]:
        """Load preset database with caching"""
        try:
            if st.session_state.preset_data is None:
                st.session_state.preset_data = FileHandler.load_preset_database()
            return st.session_state.preset_data
        except Exception as e:
            st.error(f"Error loading preset database: {str(e)}")
            return None

def main():
    """Main entry point"""
    try:
        app = PresetComparisonApp()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()