import os
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console

from rag_cli import run_query, default_persist_dir, extract_relevant_excerpt, get_date_range_from_db

console = Console()


# Page configuration
st.set_page_config(
    page_title="CEO Forum AI Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better chat appearance
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .snippet-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .ceo-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metadata {
        font-size: 0.9rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    # Use LangChain's ChatMessageHistory to keep conversation context
    st.session_state.memory = ChatMessageHistory()

if "settings" not in st.session_state:
    st.session_state.settings = {
        "persist_dir": default_persist_dir(),
        "top_k": 5,
        "compose": True,
        "max_history_turns": 20,  # Keep last 20 exchanges (40 messages)
    }

# Initialize date range (fetch once and cache)
if "date_range" not in st.session_state:
    # Get date range from database
    min_date, max_date = get_date_range_from_db(st.session_state.settings["persist_dir"])
    st.session_state.date_range = {
        "min": min_date,
        "max": max_date,
        "available": min_date is not None and max_date is not None
    }

if "date_filter" not in st.session_state:
    # Initialize date filter to full range
    st.session_state.date_filter = {
        "enabled": False,
        "min": st.session_state.date_range.get("min"),
        "max": st.session_state.date_range.get("max")
    }


def trim_conversation_memory() -> None:
    """
    Automatically trim conversation memory to prevent unbounded growth.
    Keeps only the most recent exchanges based on max_history_turns setting.
    This ensures the app won't overflow memory for non-technical users.
    """
    max_turns = st.session_state.settings.get("max_history_turns", 20)
    max_messages = max_turns * 2  # Each turn = 1 user + 1 assistant message
    
    # Trim chat history
    if len(st.session_state.chat_history) > max_messages:
        # Keep only the most recent messages
        st.session_state.chat_history = st.session_state.chat_history[-max_messages:]
    
    # Trim LangChain memory
    messages = st.session_state.memory.messages
    if len(messages) > max_messages:
        # Clear and re-add only recent messages
        trimmed_messages = messages[-max_messages:]
        st.session_state.memory.clear()
        for msg in trimmed_messages:
            st.session_state.memory.messages.append(msg)


def format_conversation_context() -> str:
    """Format recent conversation history for context."""
    messages = st.session_state.memory.messages
    
    if not messages:
        return ""
    
    # Keep last 5 exchanges (10 messages max)
    recent_messages = messages[-10:]
    
    context_parts = ["Recent conversation context:"]
    for msg in recent_messages[-6:]:  # Last 6 messages for context
        if isinstance(msg, HumanMessage):
            context_parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            context_parts.append(f"Assistant: {msg.content}")
    
    return "\n".join(context_parts)


def render_snippet_in_chat(sn: Dict, idx: int, question: str) -> None:
    """Render a snippet elegantly within a chat message."""
    ceo_display = (sn.get("speaker_display") or 
                   sn.get("ceo_display") or 
                   "Unknown CEO")
    company = sn.get("company") or sn.get("speaker_org") or ""
    title = sn.get("title") or "Untitled"
    category = sn.get("category") or ""
    date = sn.get("date") or ""
    score = float(sn.get("score", 0.0))
    
    # Use pre-extracted display_excerpt if available, otherwise fall back to text
    display_text = sn.get("display_excerpt") or sn.get("text") or ""
    full_text = sn.get("full_text")
    has_full_context = full_text and full_text != display_text and len(full_text) > len(display_text) * 1.3
    
    # Create expandable snippet
    is_interview = (category.strip().lower() == "ceo interviews") if category else False
    if is_interview:
        header_main = f"**{ceo_display}**"
    else:
        header_main = f"**{title}**"

    header_parts = [f"💼 {header_main}"]
    if is_interview and company:
        header_parts.append(f"• {company}")
    if category:
        header_parts.append(f"• {category}")
    if date:
        header_parts.append(f"• 📅 {date}")
    header_line = " ".join(header_parts)

    with st.expander(header_line, expanded=(idx == 0)):
        st.markdown(f"**Source:** {title}")
        if date:
            st.caption(f"📅 {date}")
        st.write(display_text)
        
        # Show full context if available and different from display
        if has_full_context:
            with st.expander("📖 View Full Context"):
                st.write(full_text)


def augment_query_with_context(question: str) -> str:
    """Augment the user's question with conversation context if available."""
    context = format_conversation_context()
    
    if not context:
        return question
    
    # Add context as a prefix to help the RAG system understand follow-ups
    augmented = f"""{context}

Current question: {question}

Note: If this question refers to previous context (e.g., "tell me more", "what about...", "how do they..."), 
consider the conversation history above when retrieving relevant information."""
    
    return augmented


def process_query(question: str) -> Dict:
    """Process a user query with conversation context."""
    settings = st.session_state.settings
    
    # Augment query with conversation context for better follow-up handling
    augmented_question = augment_query_with_context(question)
    
    # Get date filter parameters
    date_filter = st.session_state.date_filter
    date_min = date_filter["min"] if date_filter["enabled"] else None
    date_max = date_filter["max"] if date_filter["enabled"] else None
    
    console.log("[cyan]Running RAG query (retrieval + evaluation)...[/cyan]")
    result = run_query(
        question=augmented_question,
        persist_dir=settings["persist_dir"],
        top_k=settings["top_k"],
        compose=settings["compose"],
        composition_model="gpt-5-mini",  # Mini for final answer
        evaluation_model="gpt-5-nano",  # Nano for fast intermediate evaluations
        full_turn=True,
        agent_expand=True,
        metadata_agent=True,
        date_min=date_min,
        date_max=date_max,
    )
    
    # Post-process: Extract intelligent excerpts from long snippets IN PARALLEL
    # Do this HERE (during the spinner) not during rendering
    snippets = result.get("snippets", [])
    
    # Count how many need excerpt extraction (threshold increased from 800 to 1500 for performance)
    long_snippets_indices = [(idx, s) for idx, s in enumerate(snippets) if s.get("full_text") and len(s.get("full_text", "")) > 1500]
    if long_snippets_indices:
        import time as time_module
        t_excerpt = time_module.time()
        console.log(f"[cyan]⏱️  Extracting excerpts from {len(long_snippets_indices)} snippet(s)...[/cyan]")
    
    # Helper function for excerpt extraction
    def extract_snippet_excerpt(idx_snippet_tuple):
        idx, snippet = idx_snippet_tuple
        text = snippet.get("text", "")
        full_text = snippet.get("full_text")
        
        try:
            # Use ORIGINAL question (not augmented) for excerpt extraction with nano model
            excerpt = extract_relevant_excerpt(question, full_text, model="gpt-5-nano", max_length=800)
            # Only use excerpt if it's good quality and shorter than full text
            if excerpt and len(excerpt) > 100 and len(excerpt) < len(full_text) * 0.8:
                return idx, excerpt, "success"
            else:
                # Fallback: use text field or truncate full_text
                return idx, text if text else full_text[:800], "fallback"
        except Exception as e:
            return idx, text if text else full_text[:800], "error"
    
    # Process long snippets in parallel
    if long_snippets_indices:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_idx = {executor.submit(extract_snippet_excerpt, (idx, s)): idx for idx, s in long_snippets_indices}
            
            for future in as_completed(future_to_idx):
                try:
                    idx, excerpt, status = future.result()
                    snippets[idx]["display_excerpt"] = excerpt
                    if status == "success":
                        console.log(f"[green]  ✓ Excerpt {idx+1}/{len(snippets)} extracted[/green]")
                    elif status == "fallback":
                        console.log(f"[yellow]  • Excerpt {idx+1}/{len(snippets)} using fallback[/yellow]")
                    else:
                        console.log(f"[yellow]  • Excerpt {idx+1}/{len(snippets)} failed, using fallback[/yellow]")
                except Exception as e:
                    console.log(f"[red]  ✗ Excerpt extraction error: {e}[/red]")
        
        console.log(f"[green]⏱️  Excerpt extraction complete ({time_module.time() - t_excerpt:.2f}s)[/green]")
    
    # Set display_excerpt for short snippets (no extraction needed)
    for idx, snippet in enumerate(snippets):
        if "display_excerpt" not in snippet:
            text = snippet.get("text", "")
            full_text = snippet.get("full_text")
            snippet["display_excerpt"] = text or full_text or ""
    
    console.log("[green]✓ Query processing complete, ready to display[/green]")
    return result


def display_chat_message(role: str, content: str, result: Optional[Dict] = None, question: str = "", anchor_id: Optional[str] = None):
    """Display a chat message with optional result data."""
    with st.chat_message(role):
        if anchor_id:
            st.markdown(f"<span id=\"{anchor_id}\"></span>", unsafe_allow_html=True)
        st.write(content)
        
        if result and role == "assistant":
            snippets = result.get("snippets", [])
            sources = result.get("sources", [])
            
            # Display snippets
            if snippets:
                st.markdown("---")
                st.markdown("### 📚 Sources & Evidence")
                for idx, snippet in enumerate(snippets):
                    render_snippet_in_chat(snippet, idx, question)
            
            # Display aggregated sources summary
            if sources:
                with st.expander(f"📄 View All {len(sources)} Source Document(s)"):
                    for src in sources:
                        title = src.get("title") or "Untitled"
                        guest = src.get("guest") or ""
                        category = src.get("category") or ""
                        date = src.get("date") or ""
                        num_snippets = src.get("num_snippets", 1)
                        best_score = src.get("best_score", 0.0)
                        
                        st.markdown(f"**{title}**")
                        # Build caption with date if available
                        caption_parts = []
                        if guest:
                            caption_parts.append(guest)
                        if category:
                            caption_parts.append(category)
                        if date:
                            caption_parts.append(f"📅 {date}")
                        caption_parts.append(f"{num_snippets} snippet(s)")
                        caption_parts.append(f"Best match: {best_score:.1%}")
                        st.caption(" • ".join(caption_parts))


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.session_state.settings["persist_dir"] = st.text_input(
        "Database Connection",
        value=st.session_state.settings["persist_dir"],
        help="Connected to Chroma Cloud database (ceo_forum)",
        disabled=True
    )
    
    st.session_state.settings["top_k"] = st.slider(
        "Number of Snippets",
        min_value=3,
        max_value=10,
        value=st.session_state.settings["top_k"],
        step=1,
        help="How many relevant snippets to retrieve"
    )
    
    st.session_state.settings["compose"] = st.checkbox(
        "Generate AI Answer",
        value=st.session_state.settings["compose"],
        help="Use GPT to compose a detailed answer from snippets"
    )
    
    st.markdown("---")
    
    # Date filter
    st.subheader("📅 Date Filter")
    
    if st.session_state.date_range["available"]:
        from datetime import datetime
        
        # Parse min/max dates
        min_date_str = st.session_state.date_range["min"]
        max_date_str = st.session_state.date_range["max"]
        
        min_date_obj = datetime.strptime(min_date_str, "%Y-%m-%d").date()
        max_date_obj = datetime.strptime(max_date_str, "%Y-%m-%d").date()
        
        # Enable/disable filter with clear explanation
        date_filter_enabled = st.checkbox(
            "Enable Date Filter",
            value=st.session_state.date_filter["enabled"],
            help="When disabled, ALL sources are searched (including those without dates)"
        )
        st.session_state.date_filter["enabled"] = date_filter_enabled
        
        if date_filter_enabled:
            # Convert to month-granularity boundaries
            # Get first day of min month and last day of max month
            min_month_start = min_date_obj.replace(day=1)
            
            # Get last day of max month
            if max_date_obj.month == 12:
                max_month_end = max_date_obj.replace(year=max_date_obj.year + 1, month=1, day=1)
            else:
                max_month_end = max_date_obj.replace(month=max_date_obj.month + 1, day=1)
            from datetime import timedelta
            max_month_end = max_month_end - timedelta(days=1)
            
            # Check if we need to initialize the date selection in session state
            if "date_selection" not in st.session_state:
                st.session_state.date_selection = (min_month_start, max_month_end)
            
            # Date range picker
            selected_dates = st.date_input(
                "Select Date Range",
                value=st.session_state.date_selection,
                min_value=min_month_start,
                max_value=max_month_end,
                help="Filter sources within this date range (month granularity). Sources without dates are excluded."
            )
            
            # Handle both tuple (range) and single date
            if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
                st.session_state.date_selection = selected_dates
                selected_min, selected_max = selected_dates
                # Round to month boundaries
                selected_min_month = selected_min.replace(day=1)
                if selected_max.month == 12:
                    selected_max_month_end = selected_max.replace(year=selected_max.year + 1, month=1, day=1) - timedelta(days=1)
                else:
                    selected_max_month_end = selected_max.replace(month=selected_max.month + 1, day=1) - timedelta(days=1)
                
                st.session_state.date_filter["min"] = selected_min_month.strftime("%Y-%m-%d")
                st.session_state.date_filter["max"] = selected_max_month_end.strftime("%Y-%m-%d")
                
                # Display selected range
                st.caption(f"📅 {selected_min_month.strftime('%B %Y')} - {selected_max_month_end.strftime('%B %Y')}")
            else:
                # Reset to full range if incomplete selection
                st.session_state.date_filter["min"] = min_date_str
                st.session_state.date_filter["max"] = max_date_str
            
            # Clear/Reset button
            if st.button("🔄 Reset to Full Range", use_container_width=True):
                st.session_state.date_selection = (min_month_start, max_month_end)
                st.session_state.date_filter["min"] = min_date_str
                st.session_state.date_filter["max"] = max_date_str
                st.rerun()
        else:
            # Filter disabled - ALL sources are searched
            st.session_state.date_filter["min"] = min_date_str
            st.session_state.date_filter["max"] = max_date_str
            st.info("🔍 Searching all sources (including those without dates)")
    else:
        st.info("Date information not available in database")
    
    st.markdown("---")
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "How do I merge cultures after acquiring a technology company?",
        "What hiring practices do CEOs use to build strong cultures?",
        "What are common CEO strategies for customer experience?",
        "How can I grow my digital marketing team of 100 people?",
    ]
    
    for q in sample_questions:
        if st.button(q, key=f"sample_{q[:20]}", use_container_width=True):
            # Add to chat input (will be processed in main loop)
            st.session_state.pending_question = q
    
    st.markdown("---")
    
    # Simple clear button without technical details
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.memory.clear()
        st.rerun()
    
    st.markdown("---")
    st.caption("💻 Powered by OpenAI GPT-5")


# Main chat interface
st.title("💼 CEO Forum AI Assistant")
st.caption("Ask questions and get insights from CEO interviews, articles, and summit content")
st.success("☁️ Connected to Chroma Cloud database")

# Display existing chat history
for message in st.session_state.chat_history:
    display_chat_message(
        message["role"],
        message["content"],
        message.get("result"),
        message.get("question", "")
    )

# Handle sample question button clicks
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question
    
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "question": question
    })
    
    # Display user message
    display_chat_message("user", question, question=question)
    
    # Process query
    with st.spinner("🔍 Searching knowledge base and composing answer..."):
        try:
            t0 = time.time()
            result = process_query(question)
            dt = time.time() - t0
            
            answer = result.get("answer")
            
            if not answer:
                answer = "I found some relevant information. Please see the snippets below for details."
            
            # Add to memory
            st.session_state.memory.add_user_message(question)
            st.session_state.memory.add_ai_message(answer)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "result": result,
                "question": question
            })
            
            # Automatically trim memory to prevent overflow
            trim_conversation_memory()
            
            # Display assistant response
            display_chat_message("assistant", answer, result, question, anchor_id="last_answer")
            # Scroll to the top of the synthesized answer instead of the very bottom
            st.markdown("""
<script>
  const el = document.getElementById('last_answer');
  if (el) { el.scrollIntoView({ block: 'start' }); }
</script>
""", unsafe_allow_html=True)
            
            st.caption(f"✨ Completed in {dt:.2f}s")
            
        except Exception as exc:
            error_msg = f"I encountered an error processing your question: {str(exc)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "question": question
            })
            display_chat_message("assistant", error_msg, question=question)
    
    # Do not auto-rerun; keep the current scroll position at the synthesized answer

# Chat input
if prompt := st.chat_input("Ask a question about CEO insights, strategies, and experiences..."):
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "question": prompt
    })
    
    # Display user message
    display_chat_message("user", prompt, question=prompt)
    
    # Process query
    with st.spinner("🔍 Searching knowledge base and composing answer..."):
        try:
            t0 = time.time()
            result = process_query(prompt)
            dt = time.time() - t0
            
            answer = result.get("answer")
            
            if not answer:
                answer = "I found some relevant information. Please see the snippets below for details."
            
            # Add to memory
            st.session_state.memory.add_user_message(prompt)
            st.session_state.memory.add_ai_message(answer)
            
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "result": result,
                "question": prompt
            })
            
            # Automatically trim memory to prevent overflow
            trim_conversation_memory()
            
            # Display assistant response
            display_chat_message("assistant", answer, result, prompt, anchor_id="last_answer")
            # Scroll to the top of the synthesized answer instead of the very bottom
            st.markdown("""
<script>
  const el = document.getElementById('last_answer');
  if (el) { el.scrollIntoView({ block: 'start' }); }
</script>
""", unsafe_allow_html=True)
            
            st.caption(f"✨ Completed in {dt:.2f}s")
            
        except Exception as exc:
            error_msg = f"I encountered an error processing your question: {str(exc)}"
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "question": prompt
            })
            display_chat_message("assistant", error_msg, question=prompt)
    
    # Do not auto-rerun; keep the current scroll position at the synthesized answer
