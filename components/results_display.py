"""
Results display components
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import config

class ResultsDisplay:
    """Components for displaying comparison results"""
    
    @staticmethod
    def render_results_overview(results_data: List[Dict[str, Any]]):
        """Render overview of results with key metrics"""
        if not results_data:
            st.info("No results to display")
            return
        
        df = pd.DataFrame(results_data)
        
        # Summary metrics
        ResultsDisplay._render_summary_cards(df)
        
        # Quick stats
        st.subheader("📊 Quick Statistics")
        ResultsDisplay._render_quick_stats(df)
        
        # Distribution charts
        st.subheader("📈 Distribution Analysis")
        ResultsDisplay._render_distribution_charts(df)
    
    @staticmethod
    def _render_summary_cards(df: pd.DataFrame):
        """Render summary metric cards"""
        total_inputs = len(df)
        exact_matches = len(df[df['Status'] == config.MatchStatus.EXACT_MATCH])
        similar_matches = len(df[df['Status'] == config.MatchStatus.PARTIAL_MATCH])
        not_found = len(df[df['Status'] == config.MatchStatus.NOT_FOUND])
        
        # Calculate percentages
        exact_pct = (exact_matches / total_inputs * 100) if total_inputs > 0 else 0
        similar_pct = (similar_matches / total_inputs * 100) if total_inputs > 0 else 0
        not_found_pct = (not_found / total_inputs * 100) if total_inputs > 0 else 0
        match_rate = exact_pct + similar_pct
        
        # Average similarity
        avg_similarity = df['Similarity %'].mean() if 'Similarity %' in df.columns else 0
        
        # Display in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Inputs",
                f"{total_inputs:,}",
                help="Total number of input values processed"
            )
        
        with col2:
            st.metric(
                "Exact Matches",
                exact_matches,
                delta=f"{exact_pct:.1f}%",
                delta_color="normal",
                help="Perfect matches found"
            )
        
        with col3:
            st.metric(
                "Similar Matches", 
                similar_matches,
                delta=f"{similar_pct:.1f}%",
                delta_color="normal",
                help="Similar matches above threshold"
            )
        
        with col4:
            st.metric(
                "Not Found",
                not_found,
                delta=f"{not_found_pct:.1f}%",
                delta_color="inverse",
                help="No matches found"
            )
        
        with col5:
            st.metric(
                "Match Rate",
                f"{match_rate:.1f}%",
                delta=f"Avg: {avg_similarity:.1f}%",
                help="Overall success rate"
            )
    
    @staticmethod
    def _render_quick_stats(df: pd.DataFrame):
        """Render quick statistics table"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Status breakdown
            st.write("**Match Status Breakdown:**")
            status_counts = df['Status'].value_counts()
            status_df = pd.DataFrame({
                'Status': status_counts.index,
                'Count': status_counts.values,
                'Percentage': (status_counts.values / len(df) * 100).round(1)
            })
            st.dataframe(status_df, use_container_width=True, hide_index=True)
        
        with col2:
            # Similarity ranges
            if 'Similarity %' in df.columns:
                st.write("**Similarity Score Ranges:**")
                ranges = {
                    '90-100%': len(df[df['Similarity %'] >= 90]),
                    '80-89%': len(df[(df['Similarity %'] >= 80) & (df['Similarity %'] < 90)]),
                    '75-79%': len(df[(df['Similarity %'] >= 75) & (df['Similarity %'] < 80)]),
                    'Below 75%': len(df[df['Similarity %'] < 75])
                }
                
                ranges_df = pd.DataFrame({
                    'Range': list(ranges.keys()),
                    'Count': list(ranges.values()),
                    'Percentage': [v/len(df)*100 for v in ranges.values()]
                })
                ranges_df['Percentage'] = ranges_df['Percentage'].round(1)
                st.dataframe(ranges_df, use_container_width=True, hide_index=True)
    
    @staticmethod
    def _render_distribution_charts(df: pd.DataFrame):
        """Render distribution charts"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Status distribution pie chart
            status_counts = df['Status'].value_counts()
            
            fig_pie = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Match Status Distribution",
                color=status_counts.index,
                color_discrete_map={
                    config.MatchStatus.EXACT_MATCH: config.COLORS['exact_match'],
                    config.MatchStatus.PARTIAL_MATCH: config.COLORS['partial_match'],
                    config.MatchStatus.NOT_FOUND: config.COLORS['not_found']
                }
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Similarity histogram
            if 'Similarity %' in df.columns:
                fig_hist = px.histogram(
                    df[df['Similarity %'] > 0],  # Exclude zero similarities
                    x='Similarity %',
                    nbins=20,
                    title="Similarity Score Distribution",
                    color_discrete_sequence=[config.COLORS['primary']]
                )
                fig_hist.update_layout(
                    xaxis_title="Similarity Score (%)",
                    yaxis_title="Number of Matches"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
    
    @staticmethod
    def render_detailed_results(results_data: List[Dict[str, Any]], 
                              filters: Dict[str, Any] = None):
        """Render detailed results table with filtering"""
        if not results_data:
            st.info("No detailed results to display")
            return
        
        df = pd.DataFrame(results_data)
        
        # Apply filters if provided
        if filters:
            from components.sidebar import FilterPanel
            df = FilterPanel.apply_filters(df, filters)
        
        st.subheader(f"📋 Detailed Results ({len(df)} items)")
        
        # Table display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_all = st.checkbox("Show all columns", value=False)
        
        with col2:
            page_size = st.selectbox(
                "Items per page", 
                options=[10, 25, 50, 100, len(df)],
                index=1
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort by",
                options=['Similarity %', 'Status', 'Original Input'],
                index=0
            )
        
        # Sort dataframe
        if sort_by in df.columns:
            ascending = sort_by != 'Similarity %'  # Similarity should be descending
            df = df.sort_values(sort_by, ascending=ascending)
        
        # Select columns to display
        if show_all:
            display_df = df
        else:
            core_columns = ['Composite Key', 'Original Input', 'Matched Preset Value', 'Similarity %', 'Status']
            available_columns = [col for col in core_columns if col in df.columns]
            display_df = df[available_columns]
        
        # Pagination
        total_pages = (len(df) - 1) // page_size + 1 if len(df) > 0 else 1
        current_page = st.number_input(
            f"Page (1-{total_pages})",
            min_value=1,
            max_value=total_pages,
            value=1
        ) - 1
        
        start_idx = current_page * page_size
        end_idx = min(start_idx + page_size, len(df))
        page_df = display_df.iloc[start_idx:end_idx]
        
        # Style and display table
        styled_df = ResultsDisplay._style_detailed_table(page_df)
        st.dataframe(styled_df, use_container_width=True)
        
        # Show page info
        st.caption(f"Showing items {start_idx + 1}-{end_idx} of {len(df)}")
        
        return df
    
    @staticmethod
    def _style_detailed_table(df: pd.DataFrame):
        """Apply styling to detailed results table"""
        def highlight_status(row):
            if row['Status'] == config.MatchStatus.EXACT_MATCH:
                return ['background-color: #C8E6C9'] * len(row)
            elif row['Status'] == config.MatchStatus.PARTIAL_MATCH:
                return ['background-color: #FFE0B2'] * len(row)
            elif row['Status'] == config.MatchStatus.NOT_FOUND:
                return ['background-color: #FFCDD2'] * len(row)
            return [''] * len(row)
        
        def format_similarity(val):
            if pd.isna(val) or val == 0:
                return ""
            return f"{val:.1f}%"
        
        styled = df.style.apply(highlight_status, axis=1)
        
        if 'Similarity %' in df.columns:
            styled = styled.format({'Similarity %': format_similarity})
        
        return styled
    
    @staticmethod
    def render_export_section(results_data: List[Dict[str, Any]]):
        """Render export options section"""
        if not results_data:
            return
        
        st.subheader("📤 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Standard Export:**")
            if st.button("📊 Download Excel", use_container_width=True):
                from utils.export_manager import ExportManager
                excel_data = ExportManager.create_results_excel(results_data)
                
                st.download_button(
                    label="📥 Download Excel File",
                    data=excel_data,
                    file_name=f"comparison_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col2:
            st.write("**CSV Export:**")
            if st.button("📄 Download CSV", use_container_width=True):
                from utils.export_manager import ExportManager
                csv_data = ExportManager.create_csv_export(results_data)
                
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv_data,
                    file_name=f"comparison_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            st.write("**Custom Export:**")
            with st.expander("Filter & Export"):
                # Export filters
                export_status = st.selectbox(
                    "Export Status",
                    options=["All", config.MatchStatus.EXACT_MATCH, 
                            config.MatchStatus.PARTIAL_MATCH, config.MatchStatus.NOT_FOUND]
                )
                
                min_similarity = st.slider("Min Similarity %", 0, 100, 0)
                
                if st.button("📊 Export Filtered"):
                    filter_criteria = {
                        'status': export_status if export_status != "All" else None,
                        'min_similarity': min_similarity
                    }
                    
                    from utils.export_manager import ExportManager
                    filtered_data = ExportManager.create_filtered_export(results_data, filter_criteria)
                    
                    st.download_button(
                        label="📥 Download Filtered Excel",
                        data=filtered_data,
                        file_name=f"filtered_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    
    @staticmethod
    def render_insights_section(results_data: List[Dict[str, Any]]):
        """Render insights and recommendations"""
        if not results_data:
            return
        
        st.subheader("💡 Insights & Recommendations")
        
        from utils.export_manager import ResultsAnalyzer
        analysis = ResultsAnalyzer.analyze_results(results_data)
        
        # Display insights
        if 'insights' in analysis:
            for insight in analysis['insights']:
                st.info(f"💡 {insight}")
        
        # Detailed analysis in expander
        with st.expander("📊 Detailed Analysis"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Summary Statistics:**")
                if 'similarity_stats' in analysis:
                    stats = analysis['similarity_stats']
                    st.write(f"• Average Similarity: {stats['mean']:.1f}%")
                    st.write(f"• Median Similarity: {stats['median']:.1f}%")
                    st.write(f"• Standard Deviation: {stats['std']:.1f}%")
                    st.write(f"• Range: {stats['min']:.1f}% - {stats['max']:.1f}%")
            
            with col2:
                st.write("**Data Quality Indicators:**")
                df = pd.DataFrame(results_data)
                
                # Calculate quality metrics
                exact_rate = len(df[df['Status'] == config.MatchStatus.EXACT_MATCH]) / len(df) * 100
                match_rate = len(df[df['Status'] != config.MatchStatus.NOT_FOUND]) / len(df) * 100
                avg_similarity = df['Similarity %'].mean()
                
                if exact_rate >= 50:
                    st.success(f"• High data consistency ({exact_rate:.1f}% exact matches)")
                elif exact_rate >= 25:
                    st.warning(f"• Moderate data consistency ({exact_rate:.1f}% exact matches)")
                else:
                    st.error(f"• Low data consistency ({exact_rate:.1f}% exact matches)")
                
                if match_rate >= 85:
                    st.success(f"• Excellent match coverage ({match_rate:.1f}%)")
                elif match_rate >= 70:
                    st.info(f"• Good match coverage ({match_rate:.1f}%)")
                else:
                    st.warning(f"• Consider expanding database ({match_rate:.1f}% coverage)")