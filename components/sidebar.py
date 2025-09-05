"""
Sidebar component for navigation and controls
"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import config
from utils.file_handler import FileHandler

class Sidebar:
    """Sidebar component for navigation and settings"""
    
    @staticmethod
    def render() -> Dict[str, Any]:
        """
        Render sidebar and return current state
        
        Returns:
            Dict containing sidebar state
        """
        st.sidebar.title("🔧 Control Panel")
        
        sidebar_state = {}
        
        # Navigation
        sidebar_state['page'] = Sidebar._render_navigation()
        
        # Processing options
        sidebar_state['processing_options'] = Sidebar._render_processing_options()
        
        # Database management
        sidebar_state['database_action'] = Sidebar._render_database_management()
        
        # System information
        Sidebar._render_system_info()
        
        return sidebar_state
    
    @staticmethod
    def _render_navigation() -> str:
        """Render navigation menu"""
        st.sidebar.subheader("📍 Navigation")
        
        pages = {
            "🏠 Home": "home",
            "🔍 Compare Values": "compare", 
            "📊 Results Analysis": "analysis",
            "⚙️ Settings": "settings",
            "❓ Help": "help"
        }
        
        selected_page = st.sidebar.radio(
            "Go to:",
            options=list(pages.keys()),
            index=1  # Default to Compare Values
        )
        
        return pages[selected_page]
    
    @staticmethod
    def _render_processing_options() -> Dict[str, Any]:
        """Render processing options"""
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
        
        # Maximum results per input
        options['max_results'] = st.sidebar.number_input(
            "Max Results per Input",
            min_value=1,
            max_value=20,
            value=config.MAX_RESULTS_PER_INPUT,
            help="Maximum number of matches to return per input value"
        )
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            options['exact_threshold'] = st.slider(
                "Exact Match Threshold (%)",
                min_value=90,
                max_value=100,
                value=int(config.EXACT_MATCH_THRESHOLD * 100),
                step=1,
                help="Threshold for considering a match as 'exact'"
            ) / 100.0
            
            options['enable_semantic'] = st.checkbox(
                "Enable Semantic Matching",
                value=False,
                help="Use AI-powered semantic similarity (slower but more accurate)"
            )
            
            options['detailed_comments'] = st.checkbox(
                "Detailed Comments",
                value=True,
                help="Generate detailed explanations for matches"
            )
        
        return options
    
    @staticmethod
    def _render_database_management() -> Optional[str]:
        """Render database management options"""
        st.sidebar.subheader("🗄️ Database Management")
        
        action = None
        
        # Database info
        try:
            db_info = Sidebar._get_database_info()
            st.sidebar.info(f"**Current Database**\n\n"
                          f"Records: {db_info['records']:,}\n"
                          f"Last Modified: {db_info['modified']}")
        except Exception as e:
            st.sidebar.error("Error loading database info")
        
        # Management actions
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🔄 Refresh DB", use_container_width=True):
                st.cache_data.clear()
                action = "refresh"
        
        with col2:
            if st.button("💾 Backup DB", use_container_width=True):
                action = "backup"
        
        # Update database
        st.sidebar.write("**Update Database:**")
        new_db_file = st.sidebar.file_uploader(
            "Upload new database",
            type=['xlsx'],
            help="Replace the current preset database"
        )
        
        if new_db_file and st.sidebar.button("📤 Update Database"):
            action = "update"
            st.session_state['new_db_file'] = new_db_file
        
        return action
    
    @staticmethod
    def _get_database_info() -> Dict[str, Any]:
        """Get database information"""
        try:
            df = FileHandler.load_preset_database()
            
            import os
            from datetime import datetime
            
            modified_time = os.path.getmtime(config.PRESET_DB_PATH)
            modified_date = datetime.fromtimestamp(modified_time).strftime('%Y-%m-%d %H:%M')
            
            return {
                'records': len(df),
                'modified': modified_date,
                'columns': list(df.columns)
            }
        except Exception as e:
            return {
                'records': 0,
                'modified': 'Unknown',
                'columns': []
            }
    
    @staticmethod
    def _render_system_info():
        """Render system information"""
        with st.sidebar.expander("ℹ️ System Information"):
            st.write(f"**Version:** {config.VERSION}")
            st.write(f"**Matching Threshold:** {config.MATCHING_THRESHOLD*100}%")
            st.write(f"**Max File Size:** {config.MAX_FILE_SIZE_MB}MB")
            
            # Session state info
            if hasattr(st.session_state, 'last_processing_time'):
                st.write(f"**Last Processed:** {st.session_state.last_processing_time}")
            
            # Performance info
            if 'processing_stats' in st.session_state:
                stats = st.session_state.processing_stats
                st.write(f"**Processing Time:** {stats.get('duration', 'N/A')}")
                st.write(f"**Records Processed:** {stats.get('records', 'N/A')}")

class FilterPanel:
    """Advanced filtering panel for results"""
    
    @staticmethod
    def render(results_data: list) -> Dict[str, Any]:
        """
        Render filter panel
        
        Args:
            results_data: Current results data
            
        Returns:
            Dict containing filter settings
        """
        st.subheader("🔍 Filter Results")
        
        if not results_data:
            st.info("No data to filter")
            return {}
        
        df = pd.DataFrame(results_data)
        filters = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Status filter
            status_options = ["All"] + sorted(df['Status'].unique().tolist())
            filters['status'] = st.selectbox(
                "Match Status",
                options=status_options,
                index=0
            )
        
        with col2:
            # Similarity range filter
            if 'Similarity %' in df.columns:
                min_sim, max_sim = st.slider(
                    "Similarity Range (%)",
                    min_value=0,
                    max_value=100,
                    value=(0, 100),
                    step=1
                )
                filters['similarity_range'] = (min_sim, max_sim)
        
        with col3:
            # Text search
            filters['search_text'] = st.text_input(
                "Search Text",
                placeholder="Search in values...",
                help="Search in original input and matched preset values"
            )
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters"):
            # Category filter
            if any('category' in col.lower() for col in df.columns):
                category_cols = [col for col in df.columns if 'category' in col.lower()]
                if category_cols:
                    categories = ["All"] + sorted(df[category_cols[0]].dropna().unique().tolist())
                    filters['category'] = st.selectbox("Category", categories)
            
            # Attribute filter  
            if any('attribute' in col.lower() for col in df.columns):
                attr_cols = [col for col in df.columns if 'attribute' in col.lower()]
                if attr_cols:
                    attributes = ["All"] + sorted(df[attr_cols[0]].dropna().unique().tolist())
                    filters['attribute'] = st.selectbox("Attribute", attributes)
        
        # Apply filters button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Apply Filters", use_container_width=True):
                st.rerun()
        
        return filters
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply filters to dataframe
        
        Args:
            df: DataFrame to filter
            filters: Filter settings
            
        Returns:
            Filtered DataFrame
        """
        filtered_df = df.copy()
        
        # Status filter
        if filters.get('status') and filters['status'] != "All":
            filtered_df = filtered_df[filtered_df['Status'] == filters['status']]
        
        # Similarity range filter
        if 'similarity_range' in filters and 'Similarity %' in filtered_df.columns:
            min_sim, max_sim = filters['similarity_range']
            filtered_df = filtered_df[
                (filtered_df['Similarity %'] >= min_sim) & 
                (filtered_df['Similarity %'] <= max_sim)
            ]
        
        # Text search filter
        if filters.get('search_text'):
            search_text = filters['search_text'].lower()
            mask = (
                filtered_df['Original Input'].str.lower().str.contains(search_text, na=False) |
                filtered_df['Matched Preset Value'].str.lower().str.contains(search_text, na=False)
            )
            filtered_df = filtered_df[mask]
        
        # Category filter
        if filters.get('category') and filters['category'] != "All":
            category_cols = [col for col in filtered_df.columns if 'category' in col.lower()]
            if category_cols:
                filtered_df = filtered_df[filtered_df[category_cols[0]] == filters['category']]
        
        # Attribute filter
        if filters.get('attribute') and filters['attribute'] != "All":
            attr_cols = [col for col in filtered_df.columns if 'attribute' in col.lower()]
            if attr_cols:
                filtered_df = filtered_df[filtered_df[attr_cols[0]] == filters['attribute']]
        
        return filtered_df