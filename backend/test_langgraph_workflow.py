"""
Test script for LangGraph workflow.

This script tests the complete workflow with example scenarios to ensure
all nodes are working correctly.
"""

import asyncio
from langchain_core.messages import HumanMessage
from langgraph_workflow import langgraph_app
from init import logger


async def test_simulation_workflow():
    """Test the complete simulation workflow."""
    
    print("\n" + "="*80)
    print("Testing LangGraph Workflow - Simulation Mode")
    print("="*80 + "\n")
    
    # Test session
    config = {"configurable": {"thread_id": "test_session_001"}}
    
    # Step 1: Initial request
    print("\n[Step 1] User: Create a 2-layer agricultural field simulation")
    result1 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="Create a 2-layer agricultural field simulation")]},
        config=config
    )
    
    messages1 = result1.get("messages", [])
    if messages1:
        last_msg = messages1[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    print(f"\n[State] Mode: {result1.get('mode')}")
    print(f"[State] Use case: {result1.get('use_case')}")
    print(f"[State] Use case confirmed: {result1.get('use_case_confirmed')}")
    
    # Step 2: Confirm use case
    print("\n[Step 2] User: Yes, that's correct")
    result2 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="Yes, that's correct")]},
        config=config
    )
    
    messages2 = result2.get("messages", [])
    if messages2:
        last_msg = messages2[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    print(f"\n[State] Collection stage: {result2.get('collection_stage')}")
    print(f"[State] Use case confirmed: {result2.get('use_case_confirmed')}")
    
    # Step 3: Provide model parameters
    print("\n[Step 3] User: Title is 'Farm Survey', survey length 10m, max depth 2m")
    result3 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="Title is 'Farm Survey', survey length 10m, max depth 2m")]},
        config=config
    )
    
    messages3 = result3.get("messages", [])
    if messages3:
        last_msg = messages3[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    print(f"\n[State] Model: {result3.get('model')}")
    print(f"[State] Collection stage: {result3.get('collection_stage')}")
    
    # Step 4: Provide layer count
    print("\n[Step 4] User: 2 layers")
    result4 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="2 layers")]},
        config=config
    )
    
    messages4 = result4.get("messages", [])
    if messages4:
        last_msg = messages4[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    print(f"\n[State] Number of layers: {result4.get('num_layers')}")
    print(f"[State] Current focus: {result4.get('current_focus')}")
    
    # Step 5: Layer 1 parameters
    print("\n[Step 5] User: Layer 1 - 0.5m thickness, sand texture, normal moisture")
    result5 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="Layer 1 - 0.5m thickness, sand texture, normal moisture")]},
        config=config
    )
    
    messages5 = result5.get("messages", [])
    if messages5:
        last_msg = messages5[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    layers5 = result5.get("layers", [])
    if layers5:
        print(f"\n[State] Layer 1: {layers5[0]}")
    
    # Step 6: Layer 2 parameters
    print("\n[Step 6] User: Layer 2 - 1m thickness, loam texture, wet conditions")
    result6 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="Layer 2 - 1m thickness, loam texture, wet conditions")]},
        config=config
    )
    
    messages6 = result6.get("messages", [])
    if messages6:
        last_msg = messages6[-1]
        print(f"[Response] {last_msg.content[:200]}...")
    
    layers6 = result6.get("layers", [])
    if len(layers6) >= 2:
        print(f"\n[State] Layer 2: {layers6[1]}")
    print(f"[State] Collection stage: {result6.get('collection_stage')}")
    
    # Step 7: Antenna configuration
    print("\n[Step 7] User: 400 MHz antenna")
    result7 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="400 MHz antenna")]},
        config=config
    )
    
    messages7 = result7.get("messages", [])
    if messages7:
        last_msg = messages7[-1]
        print(f"[Response] {last_msg.content[:300]}...")
    
    print(f"\n[State] Antenna: {result7.get('antenna')}")
    print(f"[State] Parameters complete: {result7.get('parameters_complete')}")
    print(f"[State] File generated: {result7.get('file_generated')}")
    
    if result7.get('file_generated'):
        print(f"\n✅ SUCCESS! File generated at: {result7.get('file_path')}")
        print(f"File content length: {len(result7.get('file_content', ''))} characters")
    else:
        print(f"\n⚠️  File not yet generated. Collection stage: {result7.get('collection_stage')}")
    
    print("\n" + "="*80)
    print("Simulation Workflow Test Complete")
    print("="*80 + "\n")
    
    return result7


async def test_rag_workflow():
    """Test the RAG workflow."""
    
    print("\n" + "="*80)
    print("Testing LangGraph Workflow - RAG Mode")
    print("="*80 + "\n")
    
    # Test session
    config = {"configurable": {"thread_id": "test_session_rag_001"}}
    
    # Step 1: Ask a question
    print("\n[Step 1] User: What is the Peplinski model?")
    result1 = await langgraph_app.ainvoke(
        {"messages": [HumanMessage(content="What is the Peplinski model?")]},
        config=config
    )
    
    messages1 = result1.get("messages", [])
    if messages1:
        last_msg = messages1[-1]
        print(f"[Response] {last_msg.content[:400]}...")
    
    print(f"\n[State] Mode: {result1.get('mode')}")
    
    print("\n" + "="*80)
    print("RAG Workflow Test Complete")
    print("="*80 + "\n")
    
    return result1


async def main():
    """Run all tests."""
    
    print("\n" + "#"*80)
    print("# LangGraph Workflow Test Suite")
    print("#"*80)
    
    try:
        # Test simulation workflow
        await test_simulation_workflow()
        
        # Test RAG workflow
        await test_rag_workflow()
        
        print("\n" + "#"*80)
        print("# All Tests Complete!")
        print("#"*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

