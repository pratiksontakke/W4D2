"""
Data Models for Smart Meeting Assistant MCP Server

This module defines the database schema and data models for:
- User: Represents a user with timezone and preferences
- Meeting: Represents a scheduled meeting with participants and metadata

Key Design Decisions:
- Using SQLModel for type safety and ORM functionality
- Storing times in UTC for consistency across timezones
- Using JSON fields for flexible data (participants, preferences)
- Adding indexes for performance on common queries
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import SQLModel, Field, create_engine, Session, select
import json
import uuid
import pendulum
from pendulum import DateTime


class User(SQLModel, table=True):
    """
    User model representing a person who can participate in meetings.
    
    Fields:
    - user_id: Unique identifier (UUID4 format)
    - email: User's email address (used for identification)
    - name: Display name for the user
    - time_zone: IANA timezone identifier (e.g., 'America/New_York')
    - preferences: JSON object storing user preferences like:
      * working_hours: {"start": "09:00", "end": "17:00"}
      * working_days: [1, 2, 3, 4, 5]  # Monday=1, Sunday=7
      * buffer_time: 15  # minutes between meetings
      * max_meetings_per_day: 8
    """
    
    # Primary key - using UUID for uniqueness across distributed systems
    user_id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()))
    
    # User identification and display
    email: str = Field(index=True, unique=True)  # Indexed for fast lookups
    name: str
    
    # Timezone handling - critical for meeting scheduling
    time_zone: str = Field(default="UTC")  # IANA timezone name
    
    # Flexible preferences stored as JSON
    preferences: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps for audit trail
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def get_local_time(self, utc_time: datetime) -> DateTime:
        """
        Convert UTC time to user's local timezone.
        
        Args:
            utc_time: UTC datetime to convert
            
        Returns:
            Pendulum DateTime object in user's timezone
        """
        return pendulum.instance(utc_time).in_timezone(self.time_zone)
    
    def get_working_hours(self) -> Dict[str, str]:
        """
        Get user's working hours with defaults.
        
        Returns:
            Dict with 'start' and 'end' times in HH:MM format
        """
        return self.preferences.get("working_hours", {"start": "09:00", "end": "17:00"})
    
    def get_working_days(self) -> List[int]:
        """
        Get user's working days with defaults.
        
        Returns:
            List of integers (1=Monday, 7=Sunday)
        """
        return self.preferences.get("working_days", [1, 2, 3, 4, 5])  # Mon-Fri default


class Meeting(SQLModel, table=True):
    """
    Meeting model representing a scheduled meeting.
    
    Fields:
    - meeting_id: Unique identifier
    - title: Meeting title/subject
    - start_time_utc: Meeting start time in UTC
    - end_time_utc: Meeting end time in UTC
    - participants: List of user_ids attending the meeting
    - agenda: Meeting agenda text
    - effectiveness_score: AI-calculated score (0-100)
    - meeting_type: Type of meeting (standup, review, planning, etc.)
    - status: Current status (scheduled, completed, cancelled)
    """
    
    # Primary key
    meeting_id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()))
    
    # Meeting basic info
    title: str
    description: Optional[str] = None
    
    # Time fields - ALWAYS stored in UTC for consistency
    start_time_utc: datetime = Field(index=True)  # Indexed for time-based queries
    end_time_utc: datetime = Field(index=True)
    
    # Participants - stored as JSON array of user_ids
    participants: List[str] = Field(default_factory=list)
    
    # Meeting content
    agenda: Optional[str] = None
    notes: Optional[str] = None
    
    # AI-generated metrics
    effectiveness_score: Optional[float] = Field(default=None, ge=0, le=100)
    
    # Meeting metadata
    meeting_type: str = Field(default="general")  # standup, review, planning, etc.
    status: str = Field(default="scheduled")  # scheduled, completed, cancelled
    
    # Location info (can be physical or virtual)
    location: Optional[str] = None
    meeting_url: Optional[str] = None
    
    # Audit fields
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None  # user_id of meeting creator

    @property
    def duration_minutes(self) -> int:
        """
        Calculate meeting duration in minutes.
        
        Returns:
            Duration in minutes
        """
        delta = self.end_time_utc - self.start_time_utc
        return int(delta.total_seconds() / 60)
    
    def overlaps_with(self, other_start: datetime, other_end: datetime) -> bool:
        """
        Check if this meeting overlaps with another time period.
        
        Args:
            other_start: Start time of other period (UTC)
            other_end: End time of other period (UTC)
            
        Returns:
            True if there's any overlap
        """
        # Two time periods overlap if:
        # (start1 <= end2) and (start2 <= end1)
        return (self.start_time_utc <= other_end) and (other_start <= self.end_time_utc)
    
    def has_participant(self, user_id: str) -> bool:
        """
        Check if a user is a participant in this meeting.
        
        Args:
            user_id: User ID to check
            
        Returns:
            True if user is a participant
        """
        return user_id in self.participants
    
    def get_time_in_timezone(self, timezone: str) -> Dict[str, str]:
        """
        Get meeting times formatted for a specific timezone.
        
        Args:
            timezone: IANA timezone name
            
        Returns:
            Dict with formatted start and end times
        """
        start_local = pendulum.instance(self.start_time_utc).in_timezone(timezone)
        end_local = pendulum.instance(self.end_time_utc).in_timezone(timezone)
        
        return {
            "start": start_local.format("YYYY-MM-DD HH:mm:ss zz"),
            "end": end_local.format("YYYY-MM-DD HH:mm:ss zz"),
            "date": start_local.format("YYYY-MM-DD"),
            "start_time": start_local.format("HH:mm"),
            "end_time": end_local.format("HH:mm")
        }


class DatabaseManager:
    """
    Database manager for handling SQLite operations.
    
    This class provides:
    - Database initialization and schema creation
    - Session management for database operations
    - Common query methods
    - Database health checks
    """
    
    def __init__(self, database_url: str = "sqlite:///./data/meetings.db"):
        """
        Initialize database manager.
        
        Args:
            database_url: SQLite database URL
        """
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)  # Set echo=True for SQL debugging
        
    def create_tables(self):
        """
        Create all database tables.
        This is idempotent - safe to call multiple times.
        """
        SQLModel.metadata.create_all(self.engine)
        
    def get_session(self) -> Session:
        """
        Get a new database session.
        
        Returns:
            SQLModel Session object
        """
        return Session(self.engine)
    
    def health_check(self) -> bool:
        """
        Check if database is accessible.
        
        Returns:
            True if database is healthy
        """
        try:
            with self.get_session() as session:
                # Try a simple query
                session.exec(select(User).limit(1))
                return True
        except Exception:
            return False
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Find user by email address.
        
        Args:
            email: User's email address
            
        Returns:
            User object or None if not found
        """
        with self.get_session() as session:
            statement = select(User).where(User.email == email)
            return session.exec(statement).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """
        Find user by ID.
        
        Args:
            user_id: User's unique identifier
            
        Returns:
            User object or None if not found
        """
        with self.get_session() as session:
            return session.get(User, user_id)
    
    def get_meetings_for_user(self, user_id: str, start_date: datetime, end_date: datetime) -> List[Meeting]:
        """
        Get all meetings for a user within a date range.
        
        Args:
            user_id: User's unique identifier
            start_date: Start of date range (UTC)
            end_date: End of date range (UTC)
            
        Returns:
            List of Meeting objects
        """
        with self.get_session() as session:
            statement = select(Meeting).where(
                Meeting.participants.contains(user_id),
                Meeting.start_time_utc >= start_date,
                Meeting.end_time_utc <= end_date
            )
            return list(session.exec(statement))
    
    def find_conflicting_meetings(self, user_id: str, start_time: datetime, end_time: datetime) -> List[Meeting]:
        """
        Find meetings that conflict with a proposed time slot for a user.
        
        Args:
            user_id: User's unique identifier
            start_time: Proposed start time (UTC)
            end_time: Proposed end time (UTC)
            
        Returns:
            List of conflicting Meeting objects
        """
        with self.get_session() as session:
            statement = select(Meeting).where(
                Meeting.participants.contains(user_id),
                Meeting.start_time_utc < end_time,
                Meeting.end_time_utc > start_time,
                Meeting.status == "scheduled"
            )
            return list(session.exec(statement))


# Global database manager instance
# This will be initialized when the module is imported
db_manager = DatabaseManager() 