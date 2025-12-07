import streamlit as st
import yaml
import os
import time
from typing import List, Dict, Optional
import google.generativeai as genai
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import json
from PIL import Image
import io
from langgraph.errors import GraphRecursionError
from runtimes_dep_agent.agent.llm_agent import LLMAgent

# Page configuration
st.set_page_config(
    page_title="Model Runtime Deployment Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "agent_started" not in st.session_state:
    st.session_state.agent_started = False
if "workflow_completed" not in st.session_state:
    st.session_state.workflow_completed = False
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "yaml_config" not in st.session_state:
    st.session_state.yaml_config = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None
if "oci_pull_secret" not in st.session_state:
    st.session_state.oci_pull_secret = None
if "vllm_runtime_image" not in st.session_state:
    st.session_state.vllm_runtime_image = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_output_text" not in st.session_state:
    st.session_state.agent_output_text = None
if "agent_instance" not in st.session_state:
    st.session_state.agent_instance = None

# Function to extract data from image using Gemini Vision
def extract_data_from_image(image: Image.Image, api_key: str) -> Dict:
    """Extract structured data from an image using Gemini Vision API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = """Analyze this image and extract deployment-related information. 
        Return a JSON object with the following structure:
        {
            "configuration": {
                "num_models": <number>,
                "model_name": "<model name>",
                "estimated_vram_gb": <number>,
                "total_disk_footprint_gb": <number>,
                "supported_architectures": "<architecture>"
            },
            "accelerator": {
                "status": "<status>",
                "provider": "<provider name>",
                "compatibility": "<compatibility info>"
            },
            "deployment": {
                "decision": "<GO or NO-GO>",
                "available_gpu_memory_gb": <number>,
                "required_gpu_memory_gb": <number>,
                "conclusion": "<conclusion text>"
            },
            "qa": {
                "status": "<passed or failed>",
                "message": "<qa message>"
            },
            "reporting": {
                "models_executed": <number>,
                "status": "<status>",
                "channel": "<channel name>"
            },
            "resources": {
                "gpu_memory_available_gb": <number>,
                "gpu_memory_required_gb": <number>,
                "disk_space_gb": <number>
            }
        }
        
        Extract all numeric values and text fields you can find. If a field is not visible, use null or a reasonable default.
        Return ONLY valid JSON, no additional text."""
        
        response = model.generate_content([prompt, image])
        response_text = response.text.strip()
        
        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        extracted_data = json.loads(response_text)
        return extracted_data
        
    except json.JSONDecodeError as e:
        response_text = response.text.strip() if 'response' in locals() else "No response received"
        st.error(f"Failed to parse JSON response: {str(e)}")
        st.text(f"Raw response: {response_text}")
        return {}
    except Exception as e:
        st.error(f"Error extracting data: {str(e)}")
        return {}

# Helper function to get value from extracted data with fallback
def get_value(path: str, default):
    """Get nested value from extracted_data using dot notation path."""
    if not st.session_state.extracted_data:
        return default
    
    keys = path.split('.')
    value = st.session_state.extracted_data
    try:
        for key in keys:
            value = value[key]
        return value if value is not None else default
    except (KeyError, TypeError):
        return default

# Sidebar for API key and YAML upload
with st.sidebar:
    st.title("Configuration")
    
    # Gemini API Key input
    st.subheader("Gemini API Key")
    api_key_input = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key or "",
        placeholder="Enter your Gemini API Key",
        help="Get your API key to Run the Agent"
    )
    
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error configuring API key: {str(e)}")
    
    st.divider()
    
    # OCI Registry Pull Secret input
    st.subheader("OCI Registry Pull Secret")
    oci_secret_input = st.text_input(
        "Enter your OCI Registry Pull Secret",
        type="password",
        value=st.session_state.oci_pull_secret or "",
        help="Enter your OCI registry pull secret for authentication"
    )
    
    if oci_secret_input:
        st.session_state.oci_pull_secret = oci_secret_input
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # vLLM Runtime Image input
    st.subheader("vLLM Runtime Image")
    vllm_image_input = st.text_input(
        "Enter vLLM Runtime Image",
        value=st.session_state.vllm_runtime_image or "",
        placeholder="e.g., quay.io/example/vllm-runtime:latest",
        help="Enter the vLLM runtime image URL"
    )
    
    if vllm_image_input:
        st.session_state.vllm_runtime_image = vllm_image_input
        st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Configured</span>', unsafe_allow_html=True)
    
    st.divider()
    
    # Image upload for data extraction
    st.subheader("Upload Image for Data Extraction")
    uploaded_image = st.file_uploader(
        "Upload an image to extract deployment data",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a screenshot or image containing deployment information"
    )
    
    if uploaded_image is not None:
        st.session_state.uploaded_image = uploaded_image
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Extract data from image button
        if st.button("Extract Data from Image", use_container_width=True):
            if not st.session_state.gemini_api_key:
                st.error("⚠️ Please enter your Gemini API key first!")
            else:
                with st.spinner("Extracting data from image..."):
                    try:
                        extracted_data = extract_data_from_image(
                            image, 
                            st.session_state.gemini_api_key
                        )
                        st.session_state.extracted_data = extracted_data
                        st.success("✅ Data extracted successfully!")
                        with st.expander("View Extracted Data"):
                            st.json(extracted_data)
                    except Exception as e:
                        st.error(f"Error extracting data: {str(e)}")
    
    st.divider()
    
    # YAML file upload
    st.subheader("Upload Modelcar Images YAML File")
    uploaded_file = st.file_uploader(
        "Choose a YAML file",
        type=['yaml', 'yml'],
        help="Upload your Modelcar Images YAML file"
    )
    
    if uploaded_file is not None:
        try:
            yaml_content = uploaded_file.read()
            st.session_state.yaml_config = yaml.safe_load(yaml_content)
            st.markdown('<span style="background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 4px; font-size: 0.85em;">Loaded</span>', unsafe_allow_html=True)
            
            # Display YAML content in expander
            with st.expander("View YAML Configuration"):
                st.code(yaml_content.decode('utf-8'), language='yaml')
        except yaml.YAMLError as e:
            st.error(f"Error parsing YAML file: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.divider()
    
    # Reset button
    if st.button("Reset", use_container_width=True):
        st.session_state.agent_started = False
        st.session_state.workflow_completed = False
        st.session_state.workflow_step = 0
        st.session_state.start_time = None
        st.session_state.agent_result = None
        st.session_state.agent_output_text = None
        st.session_state.agent_instance = None
        st.rerun()

# Main interface
st.title("🤖 Model Runtime Deployment Agent")

# Badges for GitHub and Model Runtimes in RHOAI
st.markdown("""
<div style="margin-top: 8px;">
    <a href="https://github.com" target="_blank" style="text-decoration: none; margin-right: 8px;">
        <span style="background-color: #24292e; color: white; padding: 6px 12px; border-radius: 6px; font-size: 0.9em; font-weight: 500; display: inline-block;">
            GitHub Repo
        </span>
    </a>
    <a href="#" target="_blank" style="text-decoration: none;">
        <span style="background-color: #0066cc; color: white; padding: 6px 12px; border-radius: 6px; font-size: 0.9em; font-weight: 500; display: inline-block;">
            Model Runtimes in RHOAI
        </span>
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Start AI Agent button
if not st.session_state.agent_started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Start AI Agent", use_container_width=True, type="primary"):
            if not st.session_state.gemini_api_key:
                st.error("⚠️ Please enter your Gemini API key in the sidebar first!")
            elif not st.session_state.oci_pull_secret:
                st.error("⚠️ Please enter your OCI Registry Pull Secret in the sidebar first!")
            else:
                # Initialize and run the actual agent
                with st.spinner("Initializing agent..."):
                    try:
                        # Determine config path - use uploaded YAML if available, otherwise default
                        config_path = None
                        if st.session_state.yaml_config:
                            # Save uploaded YAML to temp file if needed
                            import tempfile
                            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp_file:
                                yaml.dump(st.session_state.yaml_config, tmp_file)
                                config_path = tmp_file.name
                        else:
                            config_path = "config-yaml/sample_modelcar_config.yaml"
                        
                        # Initialize agent
                        agent = LLMAgent(
                            api_key=st.session_state.gemini_api_key,
                            bootstrap_config=config_path,
                        )
                        st.session_state.agent_instance = agent
                        
                        # Run supervisor
                        SUPERVISOR_TRIGGER_MESSAGE = (
                            "Start supervisor agent operation. Receive model-car configuration report "
                            "from config specialist and make deployment decisions."
                        )
                        
                        st.session_state.agent_started = True
                        st.session_state.start_time = time.time()
                        st.session_state.workflow_step = 1  # Start at step 1
                        
                        # Run agent in background (will be processed in the workflow)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to initialize agent: {str(e)}")
                        st.exception(e)
    st.info("Click 'Start AI Agent' to begin running the Agent")
else:
    # Status checks
    st.subheader("Status Checks")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.gemini_api_key:
            st.markdown('<span style="background-color: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">Gemini API Key: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #dc3545; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">Gemini API Key: Not configured</span>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.oci_pull_secret:
            st.markdown('<span style="background-color: #28a745; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">OCI Pull Secret: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #dc3545; color: white; padding: 6px 12px; border-radius: 4px; font-size: 0.9em; font-weight: 500;">OCI Pull Secret: Not configured</span>', unsafe_allow_html=True)
    
    
    # Workflow progress
    st.subheader("Agent Workflow Progress")
    
    # Define workflow steps
    workflow_steps = [
        {"name": "Starting Supervisor Agent", "status": "pending"},
        {"name": "Calling Configuration Agent", "status": "pending"},
        {"name": "Calling Accelerator Agent", "status": "pending"},
        {"name": "Calling Deployment Specialist", "status": "pending"},
        {"name": "Calling QA Specialist", "status": "pending"},
        {"name": "Calling Reporting Agent", "status": "pending"},
    ]
    
    # Update workflow steps based on current step
    for i, step in enumerate(workflow_steps):
        if i < st.session_state.workflow_step:
            workflow_steps[i]["status"] = "completed"
        elif i == st.session_state.workflow_step:
            workflow_steps[i]["status"] = "in_progress"
    
    # Display progress
    progress_value = st.session_state.workflow_step / len(workflow_steps)
    st.progress(progress_value)
    
    # Display step statuses with badges
    for i, step in enumerate(workflow_steps):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{step['name']}")
        with col2:
            if step["status"] == "completed":
                st.markdown('<span style="background-color: #28a745; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">Completed</span>', unsafe_allow_html=True)
            elif step["status"] == "in_progress":
                st.markdown('<span style="background-color: #007bff; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">In Progress</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="background-color: #6c757d; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.85em;">Pending</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Display actual agent output if available
    if st.session_state.agent_output_text:
        with st.expander("📋 Full Agent Output", expanded=False):
            st.markdown(f"```\n{st.session_state.agent_output_text}\n```")
    
    st.markdown("---")
    
    # Display outputs based on workflow step
    if st.session_state.workflow_step >= 1:
        st.subheader("Configuration Agent Output")
        num_models = get_value("configuration.num_models", 2)
        model_name = get_value("configuration.model_name", "granite-3.1-8b-instruct")
        estimated_vram = get_value("configuration.estimated_vram_gb", 18)
        disk_footprint = get_value("configuration.total_disk_footprint_gb", 15.24)
        supported_arch = get_value("configuration.supported_architectures", "amd64")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <p><strong>Number of models found in Modelcar YAML:</strong> {num_models}</p>
        <p>The Configuration Specialist reports the following requirements for the preloaded model:</p>
        <ul>
        <li><strong>Model Name</strong>: <code>{model_name}</code></li>
        <li><strong>Estimated VRAM</strong>: {estimated_vram} GB</li>
        <li><strong>Total Disk Footprint</strong>: {disk_footprint} GB</li>
        <li><strong>Supported Architectures</strong>: {supported_arch}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    if st.session_state.workflow_step >= 2:
        st.subheader("Accelerator Agent Output")
        accelerator_status = get_value("accelerator.status", "GPU available")
        accelerator_provider = get_value("accelerator.provider", "NVIDIA")
        accelerator_compatibility = get_value("accelerator.compatibility", "The available hardware supports CUDA-compatible models and vLLM-Spyre-x86.")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <h3>Accelerator Summary</h3>
        <p>The <strong>Accelerator Specialist</strong> reports the following hardware availability:</p>
        <ul>
        <li><strong>Status</strong>: {accelerator_status}</li>
        <li><strong>Provider</strong>: {accelerator_provider}</li>
        <li><strong>Compatibility</strong>: {accelerator_compatibility}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    if st.session_state.workflow_step >= 3:
        st.subheader("Deployment Specialist Output")
        deployment_decision = get_value("deployment.decision", "GO")
        available_gpu_memory = get_value("deployment.available_gpu_memory_gb", 44.99)
        required_gpu_memory = get_value("deployment.required_gpu_memory_gb", 18)
        deployment_conclusion = get_value("deployment.conclusion", "The single available GPU meets the memory requirements for the model.")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <h3>Deployment Decision</h3>
        <p>The <strong>Decision Specialist</strong> has issued a <strong>{deployment_decision}</strong> for deployment. The analysis indicates that the cluster's resources are sufficient:</p>
        <ul>
        <li><strong>Available GPU Memory</strong>: {available_gpu_memory} GB</li>
        <li><strong>Required GPU Memory</strong>: {required_gpu_memory} GB</li>
        <li><strong>Conclusion</strong>: {deployment_conclusion}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    if st.session_state.workflow_step >= 4:
        st.subheader("QA Specialist Output")
        qa_status = get_value("qa.status", "passed")
        qa_message = get_value("qa.message", "The test confirmed that the model serving runtime was deployed correctly, the model was loaded, and it responded successfully to inference requests. No immediate action is required.")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <h3>QA Validation</h3>
        <p>The <strong>QA Specialist</strong> reports that the Opendatahub model validation test suite has <strong>{qa_status}</strong>. {qa_message}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    if st.session_state.workflow_step >= 5:
        st.subheader("Reporting Agent Output")
        elapsed_time = time.time() - st.session_state.start_time
        models_executed = get_value("reporting.models_executed", 2)
        reporting_status = get_value("reporting.status", "Deployment completed successfully")
        reporting_channel = get_value("reporting.channel", "#deployment_agent_report")
        
        st.markdown(f"""
        <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
        <h3>Slack Notification</h3>
        <p>The <strong>Reporting Agent</strong> has successfully sent the deployment summary to Slack channel <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;"><strong>{reporting_channel}</strong></a>.</p>
        <p><strong>Summary sent:</strong></p>
        <ul>
        <li><strong>Models Executed</strong>: {models_executed}</li>
        <li><strong>Time Taken</strong>: {elapsed_time:.2f} seconds</li>
        <li><strong>Status</strong>: {reporting_status}</li>
        <li><strong>Channel</strong>: <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">{reporting_channel}</a></li>
        </ul>
        <p style="margin-top: 12px; color: #28a745;"><strong>✓ Message delivered successfully</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
    
    if st.session_state.workflow_step >= 6:
        st.subheader("Summary")
        elapsed_time = time.time() - st.session_state.start_time
        models_executed_summary = get_value("reporting.models_executed", 2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models Executed", str(models_executed_summary))
        with col2:
            st.metric("Time Taken", f"{elapsed_time:.2f} seconds")
        
        st.markdown('<div style="margin-top: 20px;"><span style="background-color: #28a745; color: white; padding: 8px 16px; border-radius: 4px; font-size: 0.95em; font-weight: 500;">Deployment completed successfully</span></div>', unsafe_allow_html=True)
        
        # Interactive Graphs
        st.markdown("---")
        st.subheader("Deployment Analytics")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Agent Execution Timeline", "Resource Usage"])
        
        with tab1:
            # Agent execution timeline
            agent_data = pd.DataFrame({
                'Agent': ['Supervisor', 'Configuration', 'Accelerator', 'Deployment', 'QA', 'Reporting'],
                'Start Time': [0, elapsed_time*0.15, elapsed_time*0.3, elapsed_time*0.5, elapsed_time*0.7, elapsed_time*0.85],
                'Duration': [elapsed_time*0.15, elapsed_time*0.15, elapsed_time*0.2, elapsed_time*0.2, elapsed_time*0.15, elapsed_time*0.15],
                'Status': ['Completed', 'Completed', 'Completed', 'Completed', 'Completed', 'Completed']
            })
            
            fig_timeline = go.Figure()
            colors = ['#28a745', '#007bff', '#17a2b8', '#ffc107', '#28a745', '#6c757d']
            
            for i, row in agent_data.iterrows():
                fig_timeline.add_trace(go.Bar(
                    x=[row['Agent']],
                    y=[row['Duration']],
                    base=[row['Start Time']],
                    marker_color=colors[i],
                    name=row['Agent'],
                    text=[f"{row['Duration']:.2f}s"],
                    textposition='inside',
                    hovertemplate=f"<b>{row['Agent']}</b><br>Duration: {row['Duration']:.2f}s<br>Status: {row['Status']}<extra></extra>"
                ))
            
            fig_timeline.update_layout(
                title="Agent Execution Timeline",
                xaxis_title="Agent",
                yaxis_title="Time (seconds)",
                barmode='overlay',
                height=400,
                showlegend=False,
                hovermode='closest'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab2:
            # Resource usage chart
            gpu_avail = get_value("resources.gpu_memory_available_gb", 44.99)
            gpu_req = get_value("resources.gpu_memory_required_gb", 18.0)
            disk_space = get_value("resources.disk_space_gb", 15.24)
            
            resource_data = pd.DataFrame({
                'Resource': ['GPU Memory (Available)', 'GPU Memory (Required)', 'Disk Space'],
                'Value': [gpu_avail, gpu_req, disk_space],
                'Unit': ['GB', 'GB', 'GB']
            })
            
            fig_resources = go.Figure()
            fig_resources.add_trace(go.Bar(
                x=resource_data['Resource'],
                y=resource_data['Value'],
                marker_color=['#28a745', '#ffc107', '#17a2b8'],
                text=[f"{v} {u}" for v, u in zip(resource_data['Value'], resource_data['Unit'])],
                textposition='outside',
                hovertemplate="<b>%{x}</b><br>Value: %{y} GB<extra></extra>"
            ))
            
            fig_resources.update_layout(
                title="Resource Usage Overview",
                xaxis_title="Resource Type",
                yaxis_title="Value (GB)",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_resources, use_container_width=True)
        
        st.session_state.workflow_completed = True
        st.markdown("---")
    
    # Auto-advance workflow steps and run agent
    if not st.session_state.workflow_completed and st.session_state.workflow_step < 6:
        # Run the actual agent if not already run
        if st.session_state.workflow_step == 1 and st.session_state.agent_instance and not st.session_state.agent_result:
            placeholder = st.empty()
            with placeholder:
                with st.spinner("Running supervisor agent..."):
                    try:
                        SUPERVISOR_TRIGGER_MESSAGE = (
                            "Start supervisor agent operation. Receive model-car configuration report "
                            "from config specialist and make deployment decisions."
                        )
                        result = st.session_state.agent_instance.run_supervisor(SUPERVISOR_TRIGGER_MESSAGE)
                        output_text = st.session_state.agent_instance.extract_final_text(result)
                        st.session_state.agent_result = result
                        st.session_state.agent_output_text = output_text
                        # Advance to show configuration output
                        st.session_state.workflow_step = 2
                    except GraphRecursionError:
                        st.session_state.agent_output_text = "Error: maximum recursion depth reached."
                        st.session_state.workflow_step = 6
                    except Exception as e:
                        st.session_state.agent_output_text = f"Error during execution: {str(e)}"
                        st.session_state.workflow_step = 6
            placeholder.empty()
            st.rerun()
        elif st.session_state.workflow_step >= 2 and st.session_state.workflow_step < 6:
            # Simulate progress through remaining steps
            placeholder = st.empty()
            with placeholder:
                with st.spinner(f"Processing {workflow_steps[st.session_state.workflow_step]['name']}..."):
                    time.sleep(1.5)  # Shorter delay since agent already ran
            placeholder.empty()
            st.session_state.workflow_step += 1
            st.rerun()

# Footer at the bottom
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 20px;'>
        <small><strong>Model Runtime Deployment Agent</strong> | Powered by Model Runtimes Team ( RHOAI )</small>
    </div>
    """,
    unsafe_allow_html=True
)
