"""
MCP Tools for Smart Meeting Assistant

This module implements the 3 core MCP tools:
1. create_meeting - Schedule new meetings with conflict detection
2. find_optimal_slots - AI-powered time slot recommendations  
3. detect_scheduling_conflicts - Identify scheduling conflicts

Each tool is designed to be called by LLMs through the MCP protocol.
All tools include comprehensive error handling, input validation, and detailed responses.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pendulum
from fastmcp import FastMCP
import structlog

from models import User, Meeting, db_manager
from llm_client import llm_client

# Configure logging
logger = structlog.get_logger(__name__)

# Initialize MCP server
mcp = FastMCP("Smart Meeting Assistant")


@mcp.tool()
async def create_meeting(
    title: str,
    participants: List[str],  # List of user emails
    duration_minutes: int,
    start_time_iso: str,  # ISO 8601 format: "2024-01-15T14:00:00Z"
    description: str = "",
    meeting_type: str = "general",
    agenda: str = "",
    location: str = "",
    meeting_url: str = "",
    explain: bool = False
) -> str:
    """
    Create a new meeting with automatic conflict detection.
    
    This tool:
    1. Validates all participants exist in the system
    2. Converts the provided time to UTC for storage
    3. Checks for scheduling conflicts with existing meetings
    4. Creates the meeting if no conflicts are found
    5. Optionally provides an AI-generated explanation
    
    Args:
        title: Meeting title/subject
        participants: List of participant email addresses
        duration_minutes: Meeting duration in minutes (15-480)
        start_time_iso: Meeting start time in ISO 8601 format (UTC)
        description: Optional meeting description
        meeting_type: Type of meeting (standup, planning, review, etc.)
        agenda: Meeting agenda text
        location: Physical location or "Virtual"
        meeting_url: Video conference URL if virtual
        explain: If True, generates AI explanation of the scheduling decision
        
    Returns:
        JSON string with meeting details and status
    """
    try:
        logger.info("Creating meeting", title=title, participants=len(participants))
        
        # Input validation
        if not title or not title.strip():
            return json.dumps({"error": "Meeting title is required"})
        
        if not participants:
            return json.dumps({"error": "At least one participant is required"})
        
        if duration_minutes < 15 or duration_minutes > 480:
            return json.dumps({"error": "Duration must be between 15 and 480 minutes"})
        
        # Parse and validate start time
        try:
            start_time = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00'))
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=pendulum.UTC)
            start_time_utc = start_time.astimezone(pendulum.UTC).replace(tzinfo=None)
        except ValueError as e:
            return json.dumps({"error": f"Invalid start time format: {str(e)}"})
        
        # Calculate end time
        end_time_utc = start_time_utc + timedelta(minutes=duration_minutes)
        
        # Validate participants exist in the system
        with db_manager.get_session() as session:
            participant_users = []
            participant_ids = []
            
            for email in participants:
                user = db_manager.get_user_by_email(email.strip())
                if not user:
                    return json.dumps({"error": f"User not found: {email}"})
                participant_users.append(user)
                participant_ids.append(user.user_id)
        
        # Check for conflicts for all participants
        conflicts = []
        for user in participant_users:
            user_conflicts = db_manager.find_conflicting_meetings(
                user.user_id, start_time_utc, end_time_utc
            )
            if user_conflicts:
                conflicts.extend([
                    {
                        "user": user.name,
                        "user_email": user.email,
                        "conflicting_meeting": conflict.title,
                        "conflict_time": conflict.start_time_utc.isoformat() + "Z",
                        "conflict_duration": conflict.duration_minutes
                    }
                    for conflict in user_conflicts
                ])
        
        # If conflicts exist, return them without creating the meeting
        if conflicts:
            response = {
                "success": False,
                "error": "Scheduling conflicts detected",
                "conflicts": conflicts,
                "suggested_action": "Use find_optimal_slots to get alternative times"
            }
            return json.dumps(response, indent=2)
        
        # Create the meeting
        new_meeting = Meeting(
            title=title.strip(),
            description=description.strip(),
            start_time_utc=start_time_utc,
            end_time_utc=end_time_utc,
            participants=participant_ids,
            meeting_type=meeting_type,
            agenda=agenda.strip(),
            location=location.strip() if location else None,
            meeting_url=meeting_url.strip() if meeting_url else None,
            status="scheduled"
        )
        
        # Save to database
        with db_manager.get_session() as session:
            session.add(new_meeting)
            session.commit()
            session.refresh(new_meeting)  # Get the generated ID
        
        # Prepare response
        response = {
            "success": True,
            "meeting_id": new_meeting.meeting_id,
            "title": new_meeting.title,
            "start_time": start_time_utc.isoformat() + "Z",
            "end_time": end_time_utc.isoformat() + "Z",
            "duration_minutes": duration_minutes,
            "participants": [
                {"name": user.name, "email": user.email, "timezone": user.time_zone}
                for user in participant_users
            ],
            "meeting_type": meeting_type,
            "location": location or "Not specified",
            "meeting_url": meeting_url or "Not specified"
        }
        
        # Add AI explanation if requested
        if explain and llm_client.client:
            try:
                explanation = await llm_client.generate_text(
                    prompt=f"Explain why this meeting was successfully scheduled: {title} for {len(participants)} participants on {start_time_utc.strftime('%Y-%m-%d at %H:%M UTC')} for {duration_minutes} minutes.",
                    max_tokens=100,
                    temperature=0.5
                )
                response["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Failed to generate explanation: {e}")
        
        logger.info("Meeting created successfully", meeting_id=new_meeting.meeting_id)
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error creating meeting: {e}")
        return json.dumps({"error": f"Failed to create meeting: {str(e)}"})


@mcp.tool()
async def find_optimal_slots(
    participants: List[str],  # List of user emails
    duration_minutes: int,
    date_range_start: str,  # ISO 8601 format
    date_range_end: str,    # ISO 8601 format
    preferred_times: List[str] = None,  # Optional list of preferred hours like ["09:00", "14:00"]
    max_slots: int = 5,
    explain: bool = False
) -> str:
    """
    Find optimal meeting time slots for all participants using AI-powered analysis.
    
    This tool:
    1. Analyzes each participant's schedule and preferences
    2. Finds time slots where all participants are available
    3. Scores slots based on working hours, preferences, and conflicts
    4. Returns top-ranked time slots with explanations
    
    Args:
        participants: List of participant email addresses
        duration_minutes: Required meeting duration in minutes
        date_range_start: Start of search range (ISO 8601 format)
        date_range_end: End of search range (ISO 8601 format)  
        preferred_times: Optional list of preferred start times ["HH:MM"]
        max_slots: Maximum number of slots to return (1-10)
        explain: If True, generates AI explanations for recommendations
        
    Returns:
        JSON string with ranked time slots and analysis
    """
    try:
        logger.info("Finding optimal slots", participants=len(participants), duration=duration_minutes)
        
        # Input validation
        if not participants:
            return json.dumps({"error": "At least one participant is required"})
        
        if duration_minutes < 15 or duration_minutes > 480:
            return json.dumps({"error": "Duration must be between 15 and 480 minutes"})
        
        if max_slots < 1 or max_slots > 10:
            max_slots = 5
        
        # Parse date range
        try:
            start_date = datetime.fromisoformat(date_range_start.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(date_range_end.replace('Z', '+00:00'))
            
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=pendulum.UTC)
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=pendulum.UTC)
                
            start_date_utc = start_date.astimezone(pendulum.UTC).replace(tzinfo=None)
            end_date_utc = end_date.astimezone(pendulum.UTC).replace(tzinfo=None)
            
        except ValueError as e:
            return json.dumps({"error": f"Invalid date format: {str(e)}"})
        
        # Validate date range
        if start_date_utc >= end_date_utc:
            return json.dumps({"error": "Start date must be before end date"})
        
        if (end_date_utc - start_date_utc).days > 30:
            return json.dumps({"error": "Date range cannot exceed 30 days"})
        
        # Get participant information
        participant_users = []
        with db_manager.get_session() as session:
            for email in participants:
                user = db_manager.get_user_by_email(email.strip())
                if not user:
                    return json.dumps({"error": f"User not found: {email}"})
                participant_users.append(user)
        
        # Generate candidate time slots
        candidate_slots = []
        current_time = start_date_utc
        
        while current_time + timedelta(minutes=duration_minutes) <= end_date_utc:
            # Skip weekends for most business meetings
            if current_time.weekday() < 5:  # Monday = 0, Friday = 4
                # Generate slots at 15-minute intervals during business hours
                if 8 <= current_time.hour <= 17:  # Rough business hours in UTC
                    candidate_slots.append(current_time)
            
            current_time += timedelta(minutes=15)  # 15-minute intervals
        
        # Score each candidate slot
        scored_slots = []
        for slot_start in candidate_slots:
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            score = await _score_time_slot(slot_start, slot_end, participant_users, preferred_times)
            
            if score > 0:  # Only include viable slots
                scored_slots.append({
                    "start_time": slot_start.isoformat() + "Z",
                    "end_time": slot_end.isoformat() + "Z",
                    "score": score,
                    "conflicts": score < 100  # If score < 100, there might be minor conflicts
                })
        
        # Sort by score (highest first) and take top slots
        scored_slots.sort(key=lambda x: x["score"], reverse=True)
        top_slots = scored_slots[:max_slots]
        
        # Prepare response
        response = {
            "success": True,
            "search_parameters": {
                "participants": len(participants),
                "duration_minutes": duration_minutes,
                "date_range": f"{date_range_start} to {date_range_end}",
                "slots_analyzed": len(candidate_slots),
                "viable_slots_found": len(scored_slots)
            },
            "recommended_slots": top_slots
        }
        
        # Add AI explanations if requested
        if explain and llm_client.client and top_slots:
            try:
                for slot in top_slots:
                    explanation = await llm_client.explain_optimal_slot({
                        "time": slot["start_time"],
                        "score": slot["score"],
                        "participant_count": len(participants),
                        "conflicts": 0 if slot["score"] == 100 else 1
                    })
                    slot["explanation"] = explanation
            except Exception as e:
                logger.warning(f"Failed to generate explanations: {e}")
        
        # Add summary message
        if not top_slots:
            response["message"] = "No suitable time slots found. Try expanding the date range or reducing the duration."
        else:
            response["message"] = f"Found {len(top_slots)} optimal time slots. Highest scored slot: {top_slots[0]['score']}/100"
        
        logger.info("Optimal slots found", slots_returned=len(top_slots))
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error finding optimal slots: {e}")
        return json.dumps({"error": f"Failed to find optimal slots: {str(e)}"})


async def _score_time_slot(
    start_time: datetime, 
    end_time: datetime, 
    participants: List[User],
    preferred_times: Optional[List[str]] = None
) -> int:
    """
    Score a time slot based on participant availability and preferences.
    
    Args:
        start_time: Slot start time (UTC)
        end_time: Slot end time (UTC)
        participants: List of participant User objects
        preferred_times: Optional preferred start times
        
    Returns:
        Score from 0-100 (100 = perfect, 0 = not viable)
    """
    total_score = 0
    participant_count = len(participants)
    
    for user in participants:
        user_score = 100  # Start with perfect score
        
        # Check for conflicts
        conflicts = db_manager.find_conflicting_meetings(user.user_id, start_time, end_time)
        if conflicts:
            return 0  # Hard conflict - slot not viable
        
        # Convert to user's local time for preference checking
        user_start_local = pendulum.instance(start_time).in_timezone(user.time_zone)
        user_end_local = pendulum.instance(end_time).in_timezone(user.time_zone)
        
        # Check working days
        working_days = user.get_working_days()
        if user_start_local.weekday() + 1 not in working_days:  # Pendulum: Monday=0
            user_score -= 30  # Penalize non-working days
        
        # Check working hours
        working_hours = user.get_working_hours()
        work_start = user_start_local.replace(
            hour=int(working_hours["start"].split(":")[0]),
            minute=int(working_hours["start"].split(":")[1])
        )
        work_end = user_start_local.replace(
            hour=int(working_hours["end"].split(":")[0]),
            minute=int(working_hours["end"].split(":")[1])
        )
        
        # Penalize meetings outside working hours
        if user_start_local < work_start or user_end_local > work_end:
            user_score -= 40
        
        # Bonus for preferred times
        if preferred_times:
            start_time_str = user_start_local.format("HH:mm")
            if start_time_str in preferred_times:
                user_score += 10
        
        # Bonus for common meeting times (9 AM, 10 AM, 2 PM, 3 PM in local time)
        preferred_hours = [9, 10, 14, 15]
        if user_start_local.hour in preferred_hours:
            user_score += 5
        
        total_score += max(0, user_score)  # Ensure non-negative
    
    # Return average score across all participants
    return min(100, total_score // participant_count)


@mcp.tool()
async def detect_scheduling_conflicts(
    user_email: str,
    time_range_start: str,  # ISO 8601 format
    time_range_end: str,    # ISO 8601 format
    include_details: bool = True
) -> str:
    """
    Detect scheduling conflicts for a specific user within a time range.
    
    This tool:
    1. Finds all meetings for the user in the specified time range
    2. Identifies overlapping meetings (conflicts)
    3. Provides detailed conflict information
    4. Suggests resolution strategies
    
    Args:
        user_email: Email address of the user to check
        time_range_start: Start of time range to check (ISO 8601 format)
        time_range_end: End of time range to check (ISO 8601 format)
        include_details: If True, includes detailed conflict information
        
    Returns:
        JSON string with conflict analysis and details
    """
    try:
        logger.info("Detecting conflicts", user=user_email)
        
        # Input validation
        if not user_email or not user_email.strip():
            return json.dumps({"error": "User email is required"})
        
        # Parse time range
        try:
            start_time = datetime.fromisoformat(time_range_start.replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(time_range_end.replace('Z', '+00:00'))
            
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=pendulum.UTC)
            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=pendulum.UTC)
                
            start_time_utc = start_time.astimezone(pendulum.UTC).replace(tzinfo=None)
            end_time_utc = end_time.astimezone(pendulum.UTC).replace(tzinfo=None)
            
        except ValueError as e:
            return json.dumps({"error": f"Invalid time format: {str(e)}"})
        
        # Validate time range
        if start_time_utc >= end_time_utc:
            return json.dumps({"error": "Start time must be before end time"})
        
        # Get user
        user = db_manager.get_user_by_email(user_email.strip())
        if not user:
            return json.dumps({"error": f"User not found: {user_email}"})
        
        # Get all meetings for user in the time range
        user_meetings = db_manager.get_meetings_for_user(user.user_id, start_time_utc, end_time_utc)
        
        # Sort meetings by start time
        user_meetings.sort(key=lambda m: m.start_time_utc)
        
        # Find overlapping meetings
        conflicts = []
        for i, meeting1 in enumerate(user_meetings):
            for j, meeting2 in enumerate(user_meetings[i+1:], i+1):
                if meeting1.overlaps_with(meeting2.start_time_utc, meeting2.end_time_utc):
                    # Calculate overlap duration
                    overlap_start = max(meeting1.start_time_utc, meeting2.start_time_utc)
                    overlap_end = min(meeting1.end_time_utc, meeting2.end_time_utc)
                    overlap_minutes = int((overlap_end - overlap_start).total_seconds() / 60)
                    
                    conflict_info = {
                        "conflict_id": f"{meeting1.meeting_id}_{meeting2.meeting_id}",
                        "meeting1": {
                            "id": meeting1.meeting_id,
                            "title": meeting1.title,
                            "start_time": meeting1.start_time_utc.isoformat() + "Z",
                            "end_time": meeting1.end_time_utc.isoformat() + "Z",
                            "duration_minutes": meeting1.duration_minutes
                        },
                        "meeting2": {
                            "id": meeting2.meeting_id,
                            "title": meeting2.title,
                            "start_time": meeting2.start_time_utc.isoformat() + "Z",
                            "end_time": meeting2.end_time_utc.isoformat() + "Z",
                            "duration_minutes": meeting2.duration_minutes
                        },
                        "overlap_minutes": overlap_minutes,
                        "severity": "high" if overlap_minutes > 30 else "medium" if overlap_minutes > 15 else "low"
                    }
                    
                    if include_details:
                        # Add timezone-specific information
                        meeting1_local = meeting1.get_time_in_timezone(user.time_zone)
                        meeting2_local = meeting2.get_time_in_timezone(user.time_zone)
                        
                        conflict_info["local_times"] = {
                            "timezone": user.time_zone,
                            "meeting1": meeting1_local,
                            "meeting2": meeting2_local
                        }
                        
                        # Suggest resolution strategies
                        conflict_info["resolution_suggestions"] = [
                            f"Reschedule '{meeting1.title}' to avoid overlap",
                            f"Reschedule '{meeting2.title}' to avoid overlap", 
                            f"Shorten one meeting by {overlap_minutes} minutes",
                            "Delegate attendance if possible"
                        ]
                    
                    conflicts.append(conflict_info)
        
        # Prepare response
        response = {
            "success": True,
            "user": {
                "name": user.name,
                "email": user.email,
                "timezone": user.time_zone
            },
            "analysis_period": {
                "start": time_range_start,
                "end": time_range_end,
                "total_meetings": len(user_meetings),
                "conflicts_found": len(conflicts)
            },
            "conflicts": conflicts
        }
        
        # Add summary
        if not conflicts:
            response["summary"] = "No scheduling conflicts detected in the specified time range."
        else:
            high_severity = len([c for c in conflicts if c["severity"] == "high"])
            medium_severity = len([c for c in conflicts if c["severity"] == "medium"])
            low_severity = len([c for c in conflicts if c["severity"] == "low"])
            
            response["summary"] = f"Found {len(conflicts)} conflicts: {high_severity} high severity, {medium_severity} medium severity, {low_severity} low severity."
            response["recommendation"] = "Review conflicts and reschedule meetings to resolve overlaps."
        
        logger.info("Conflict detection completed", conflicts_found=len(conflicts))
        return json.dumps(response, indent=2)
        
    except Exception as e:
        logger.error(f"Error detecting conflicts: {e}")
        return json.dumps({"error": f"Failed to detect conflicts: {str(e)}"})


# Export the MCP server instance
__all__ = ["mcp"] 