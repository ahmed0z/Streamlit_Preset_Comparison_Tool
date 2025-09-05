"""
Reusable UI components for the Streamlit application
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import config

class UIComponents:
    """Collection of reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.set_page_config(
            page_title=config.APP_TITLE,
            page_icon=config.APP_ICON,
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title(config.APP_TITLE)
        st.markdown(f"**{config.DESCRIPTION}**")
        
        # Add version info in small text
        st.caption(f"Version {config.VERSION}")
    
    @staticmethod
    def render_file_uploader(accept_multiple: bool = False) -> Any:
        """
        Render file uploader component
        
        Args:
            accept_multiple: Whether to accept multiple files
            
        Returns:
            Uploaded file(s) or None
        """
        st.subheader("📁 File Upload")
        
        help_text = (
            "Upload an Excel file with the same structure as the template. "
            f"Maximum file size: {config.MAX_FILE_SIZE_MB}MB"
        )
        
        uploaded_file = st.file_uploader(
            "Choose your input Excel file",
            type=config.ALLOWED_EXTENSIONS,
            help=help_text,
            accept_multiple_files=accept_multiple
        )
        
        return uploaded_file
    
    @staticmethod
    def render_template_download():
        """Render template download section"""
        st.subheader("📋 Download Template")
        st.write("Download the Excel template to see the expected format:")
        
        from utils.file_handler import FileHandler
        
        try:
            template_bytes = FileHandler.create_input_template()
            
            st.download_button(
                label="📥 Download Input Template",
                data=template_bytes,
                file_name=f"input_template_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a sample template showing the required format"
            )
        except Exception as e:
            st.error(f"Error creating template: {str(e)}")
    
    @staticmethod
    def render_processing_options() -> Dict[str, Any]:
        """
        Render processing options sidebar
        
        Returns:
            Dict with processing options
        """
        st.sidebar.subheader("⚙️ Processing Options")
        
        options = {}
        
        # Matching threshold
        options['threshold'] = st.sidebar.slider(
            "Similarity Threshold (%)",
            min_value=50,
            max_value=100,
            value=int(config.MATCHING_THRESHOLD * 100),
            step=5,
            help="Minimum similarity score required for a match"
        ) / 100.0
        
        # Maximum results
        options['max_results'] = st.sidebar.number_input(
            "Max Results per Input",
            min_value=1,
            max_value=20,
            value=config.MAX_RESULTS_PER_INPUT,
            help="Maximum number of matches to return per input value"
        )
        
        # Processing mode
        options['detailed_analysis'] = st.sidebar.checkbox(
            "Detailed Analysis",
            value=True,
            help="Include detailed component analysis and statistics"
        )
        
        return options
    
    @staticmethod
    def render_results_table(results_data: List[Dict[str, Any]], 
                           show_filters: bool = True) -> pd.DataFrame:
        """
        Render interactive results table
        
        Args:
            results_data: List of result dictionaries
            show_filters: Whether to show filtering options
            
        Returns:
            Filtered DataFrame
        """
        if not results_data:
            st.info("No results to display")
            return pd.DataFrame()
        
        df = pd.DataFrame(results_data)
        
        if show_filters:
            # Filter controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                status_filter = st.selectbox(
                    "Filter by Status",
                    options=["All"] + [config.MatchStatus.EXACT_MATCH, 
                            config.MatchStatus.PARTIAL_MATCH, 
                            config.MatchStatus.NOT_FOUND],
                    index=0
                )
            
            with col2:
                min_similarity = st.slider(
                    "Minimum Similarity %",
                    min_value=0,
                    max_value=100,
                    value=0
                )
            
            with col3:
                search_term = st.text_input(
                    "Search in values",
                    placeholder="Type to search..."
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if status_filter != "All":
                filtered_df = filtered_df[filtered_df['Status'] == status_filter]
            
            if min_similarity > 0:
                filtered_df = filtered_df[filtered_df['Similarity %'] >= min_similarity]
            
            if search_term:
                mask = (filtered_df['Original Input'].str.contains(search_term, case=False, na=False) |
                       filtered_df['Matched Preset Value'].str.contains(search_term, case=False, na=False))
                filtered_df = filtered_df[mask]
        else:
            filtered_df = df
        
        # Display count
        if show_filters and len(filtered_df) != len(df):
            st.info(f"Showing {len(filtered_df)} of {len(df)} results")
        
        # Style the dataframe
        if not filtered_df.empty:
            styled_df = UIComponents._style_results_dataframe(filtered_df)
            st.dataframe(styled_df, use_container_width=True, height=400)
        else:
            st.warning("No results match the current filters")
        
        return filtered_df
    
    @staticmethod
    def _style_results_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Apply styling to results dataframe"""
        def highlight_status(val):
            if val == config.MatchStatus.EXACT_MATCH:
                return 'background-color: #C8E6C9'  # Light green
            elif val == config.MatchStatus.PARTIAL_MATCH:
                return 'background-color: #FFE0B2'  # Light orange
            elif val == config.MatchStatus.NOT_FOUND:
                return 'background-color: #FFCDD2'  # Light red
            return ''
        
        def color_similarity(val):
            try:
                val_num = float(val)
                if val_num >= 95:
                    return 'color: #2E7D32; font-weight: bold'  # Dark green
                elif val_num >= 85:
                    return 'color: #F57C00; font-weight: bold'  # Orange
                elif val_num >= 75:
                    return 'color: #E64A19'  # Red-orange
                else:
                    return 'color: #C62828'  # Red
            except:
                return ''
        
        styled = df.style.applymap(highlight_status, subset=['Status'])
        
        if 'Similarity %' in df.columns:
            styled = styled.applymap(color_similarity, subset=['Similarity %'])
        
        return styled
    
    @staticmethod
    def render_summary_metrics(results_data: List[Dict[str, Any]]):
        """Render summary metrics cards"""
        if not results_data:
            return
        
        df = pd.DataFrame(results_data)
        
        # Calculate metrics
        total_inputs = len(df)
        exact_matches = len(df[df['Status'] == config.MatchStatus.EXACT_MATCH])
        similar_matches = len(df[df['Status'] == config.MatchStatus.PARTIAL_MATCH])
        not_found = len(df[df['Status'] == config.MatchStatus.NOT_FOUND])
        avg_similarity = df['Similarity %'].mean() if 'Similarity %' in df.columns else 0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Inputs", total_inputs)
        
        with col2:
            st.metric("Exact Matches", exact_matches, 
                     delta=f"{(exact_matches/total_inputs*100):.1f}%" if total_inputs > 0 else None)
        
        with col3:
            st.metric("Similar Matches", similar_matches,
                     delta=f"{(similar_matches/total_inputs*100):.1f}%" if total_inputs > 0 else None)
        
        with col4:
            st.metric("Not Found", not_found,
                     delta=f"{(not_found/total_inputs*100):.1f}%" if total_inputs > 0 else None)
        
        with col5:
            st.metric("Avg Similarity", f"{avg_similarity:.1f}%")
    
    @staticmethod
    def render_results_charts(results_data: List[Dict[str, Any]]):
        """Render charts for results analysis"""
        if not results_data:
            return
        
        df = pd.DataFrame(results_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution pie chart
            st.subheader("Match Status Distribution")
            status_counts = df['Status'].value_counts()
            
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                color=status_counts.index,
                color_discrete_map={
                    config.MatchStatus.EXACT_MATCH: config.COLORS['exact_match'],
                    config.MatchStatus.PARTIAL_MATCH: config.COLORS['partial_match'],
                    config.MatchStatus.NOT_FOUND: config.COLORS['not_found']
                }
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Similarity score distribution
            if 'Similarity %' in df.columns:
                st.subheader("Similarity Score Distribution")
                
                fig_hist = px.histogram(
                    df,
                    x='Similarity %',
                    nbins=20,
                    title="Distribution of Similarity Scores",
                    color_discrete_sequence=[config.COLORS['primary']]
                )
                fig_hist.update_layout(
                    xaxis_title="Similarity %",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def render_export_options(results_data: List[Dict[str, Any]]) -> Optional[str]:
        """
        Render export options and return selected format
        
        Args:
            results_data: Results to export
            
        Returns:
            Selected export format or None
        """
        if not results_data:
            return None
        
        st.subheader("📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        export_format = None
        
        with col1:
            if st.button("📊 Export to Excel", use_container_width=True):
                export_format = "excel"
        
        with col2:
            if st.button("📄 Export to CSV", use_container_width=True):
                export_format = "csv"
        
        with col3:
            if st.button("🔍 Filtered Export", use_container_width=True):
                export_format = "filtered"
        
        return export_format
    
    @staticmethod
    def render_progress_bar(progress: float, text: str = "Processing..."):
        """Render progress bar"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        progress_bar.progress(progress)
        status_text.text(text)
        
        return progress_bar, status_text
    
    @staticmethod
    def render_error_message(error: str, details: str = None):
        """Render error message with optional details"""
        st.error(f"❌ {error}")
        
        if details:
            with st.expander("Error Details"):
                st.code(details)
    
    @staticmethod
    def render_success_message(message: str, details: str = None):
        """Render success message with optional details"""
        st.success(f"✅ {message}")
        
        if details:
            st.info(details)
    
    @staticmethod
    def render_info_box(title: str, content: str, type: str = "info"):
        """Render information box"""
        if type == "info":
            st.info(f"**{title}**\n\n{content}")
        elif type == "warning":
            st.warning(f"**{title}**\n\n{content}")
        elif type == "success":
            st.success(f"**{title}**\n\n{content}")
        elif type == "error":
            st.error(f"**{title}**\n\n{content}")
    
    @staticmethod
    def render_help_section():
        """Render help and instructions section"""
        with st.expander("❓ Help & Instructions"):
            st.markdown("""
            ### How to Use This Tool
            
            1. **Download Template**: Click the download button to get the Excel template
            2. **Fill Your Data**: Replace sample data with your actual values
            3. **Upload File**: Use the file uploader to select your completed template
            4. **Review Results**: Examine the comparison results and statistics
            5. **Export**: Download the results in Excel or CSV format
            
            ### Supported Value Formats
            
            - **Plain text**: "Protective Cap"
            - **Numbers with units**: "20 kg", "0.181 inches"
            - **Multiple values**: "20 kg, 15 cm"
            - **Values with conditions**: "20 kg @ 30°C"
            - **Complex dimensions**: "L x W x H (mm x mm x mm)"
            
            ### Match Types
            
            - **Exact Match**: Perfect match found (100% similarity)
            - **Partial Match**: High similarity found (≥75% similarity)
            - **Not Found**: No suitable match found (<75% similarity)
            
            ### Tips for Better Results
            
            - Use consistent formatting in your input data
            - Include units when applicable
            - Check spelling and terminology
            - Review "Not Found" items for potential data quality issues
            """)