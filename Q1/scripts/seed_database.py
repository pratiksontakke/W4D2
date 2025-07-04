#!/usr/bin/env python3
"""
Database Seeding Script for Smart Meeting Assistant

This script creates sample data for testing the MCP server:
- 8 users across different timezones
- 70+ meetings with realistic scheduling patterns
- Various meeting types and durations
- Realistic conflict scenarios for testing

Run this script to populate the database with test data.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
import random
from typing import List, Dict

# Add src directory to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import User, Meeting, DatabaseManager, db_manager
import pendulum

# Sample data for realistic meetings
SAMPLE_USERS = [
    {
        "email": "alice.johnson@company.com",
        "name": "Alice Johnson",
        "time_zone": "America/New_York",
        "preferences": {
            "working_hours": {"start": "09:00", "end": "17:00"},
            "working_days": [1, 2, 3, 4, 5],  # Mon-Fri
            "buffer_time": 15,
            "max_meetings_per_day": 6
        }
    },
    {
        "email": "bob.smith@company.com", 
        "name": "Bob Smith",
        "time_zone": "America/Los_Angeles",
        "preferences": {
            "working_hours": {"start": "08:00", "end": "16:00"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 10,
            "max_meetings_per_day": 8
        }
    },
    {
        "email": "carol.davis@company.com",
        "name": "Carol Davis", 
        "time_zone": "Europe/London",
        "preferences": {
            "working_hours": {"start": "09:30", "end": "17:30"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 15,
            "max_meetings_per_day": 5
        }
    },
    {
        "email": "david.wilson@company.com",
        "name": "David Wilson",
        "time_zone": "Asia/Tokyo", 
        "preferences": {
            "working_hours": {"start": "09:00", "end": "18:00"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 20,
            "max_meetings_per_day": 7
        }
    },
    {
        "email": "emma.brown@company.com",
        "name": "Emma Brown",
        "time_zone": "Australia/Sydney",
        "preferences": {
            "working_hours": {"start": "08:30", "end": "16:30"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 15,
            "max_meetings_per_day": 6
        }
    },
    {
        "email": "frank.miller@company.com",
        "name": "Frank Miller",
        "time_zone": "America/Chicago",
        "preferences": {
            "working_hours": {"start": "08:00", "end": "17:00"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 10,
            "max_meetings_per_day": 8
        }
    },
    {
        "email": "grace.lee@company.com",
        "name": "Grace Lee",
        "time_zone": "Europe/Berlin",
        "preferences": {
            "working_hours": {"start": "09:00", "end": "17:00"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 15,
            "max_meetings_per_day": 5
        }
    },
    {
        "email": "henry.garcia@company.com",
        "name": "Henry Garcia",
        "time_zone": "America/Denver",
        "preferences": {
            "working_hours": {"start": "07:30", "end": "15:30"},
            "working_days": [1, 2, 3, 4, 5],
            "buffer_time": 20,
            "max_meetings_per_day": 6
        }
    }
]

# Meeting templates for realistic scenarios
MEETING_TEMPLATES = [
    {
        "title": "Weekly Team Standup",
        "type": "standup",
        "duration": 30,
        "frequency": "weekly",
        "participant_count": 4
    },
    {
        "title": "Product Planning Session",
        "type": "planning", 
        "duration": 90,
        "frequency": "biweekly",
        "participant_count": 6
    },
    {
        "title": "Code Review Meeting",
        "type": "review",
        "duration": 45,
        "frequency": "weekly",
        "participant_count": 3
    },
    {
        "title": "Client Presentation",
        "type": "presentation",
        "duration": 60,
        "frequency": "monthly",
        "participant_count": 5
    },
    {
        "title": "Sprint Retrospective",
        "type": "retrospective",
        "duration": 60,
        "frequency": "biweekly", 
        "participant_count": 5
    },
    {
        "title": "Architecture Discussion",
        "type": "discussion",
        "duration": 120,
        "frequency": "monthly",
        "participant_count": 4
    },
    {
        "title": "1:1 Check-in",
        "type": "one_on_one",
        "duration": 30,
        "frequency": "weekly",
        "participant_count": 2
    },
    {
        "title": "Marketing Strategy Meeting",
        "type": "strategy",
        "duration": 90,
        "frequency": "monthly",
        "participant_count": 6
    },
    {
        "title": "Bug Triage",
        "type": "triage",
        "duration": 45,
        "frequency": "weekly",
        "participant_count": 4
    },
    {
        "title": "All Hands Meeting",
        "type": "all_hands",
        "duration": 60,
        "frequency": "monthly",
        "participant_count": 8
    }
]

def create_users() -> List[User]:
    """
    Create user objects from sample data.
    
    Returns:
        List of User objects
    """
    users = []
    for user_data in SAMPLE_USERS:
        user = User(
            email=user_data["email"],
            name=user_data["name"],
            time_zone=user_data["time_zone"],
            preferences=user_data["preferences"]
        )
        users.append(user)
    return users

def generate_meeting_times(start_date: datetime, days: int = 30) -> List[datetime]:
    """
    Generate realistic meeting times over a date range.
    
    Args:
        start_date: Starting date for meeting generation
        days: Number of days to generate meetings for
        
    Returns:
        List of datetime objects representing meeting start times
    """
    meeting_times = []
    current_date = start_date
    
    for day in range(days):
        # Skip weekends for most meetings
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            # Generate 2-4 meetings per day
            meetings_today = random.randint(2, 4)
            
            for _ in range(meetings_today):
                # Meeting times between 9 AM and 5 PM
                hour = random.randint(9, 16)
                minute = random.choice([0, 15, 30, 45])  # Common meeting times
                
                meeting_time = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                meeting_times.append(meeting_time)
        
        current_date += timedelta(days=1)
    
    return sorted(meeting_times)

def create_meetings(users: List[User], meeting_times: List[datetime]) -> List[Meeting]:
    """
    Create meeting objects with realistic participant combinations.
    
    Args:
        users: List of User objects
        meeting_times: List of meeting start times
        
    Returns:
        List of Meeting objects
    """
    meetings = []
    user_ids = [user.user_id for user in users]
    
    for i, start_time in enumerate(meeting_times):
        # Choose a random meeting template
        template = random.choice(MEETING_TEMPLATES)
        
        # Calculate end time
        end_time = start_time + timedelta(minutes=template["duration"])
        
        # Select participants
        participant_count = min(template["participant_count"], len(users))
        participants = random.sample(user_ids, participant_count)
        
        # Create meeting with some variation in titles
        title_variations = [
            template["title"],
            f"{template['title']} - Week {i//7 + 1}",
            f"{template['title']} (Team Alpha)",
            f"{template['title']} - Q1 Planning"
        ]
        
        meeting = Meeting(
            title=random.choice(title_variations),
            description=f"Scheduled {template['type']} meeting",
            start_time_utc=start_time,
            end_time_utc=end_time,
            participants=participants,
            meeting_type=template["type"],
            status="scheduled",
            agenda=generate_sample_agenda(template["type"]),
            location="Conference Room A" if random.random() > 0.5 else None,
            meeting_url="https://meet.company.com/room123" if random.random() > 0.3 else None
        )
        
        meetings.append(meeting)
    
    return meetings

def generate_sample_agenda(meeting_type: str) -> str:
    """
    Generate sample agenda based on meeting type.
    
    Args:
        meeting_type: Type of meeting
        
    Returns:
        Sample agenda text
    """
    agendas = {
        "standup": "1. What did you work on yesterday?\n2. What will you work on today?\n3. Any blockers?",
        "planning": "1. Review previous sprint\n2. Discuss upcoming features\n3. Estimate story points\n4. Set sprint goals",
        "review": "1. Code review guidelines\n2. Review pending PRs\n3. Discuss best practices\n4. Action items",
        "presentation": "1. Project overview\n2. Demo\n3. Q&A session\n4. Next steps",
        "retrospective": "1. What went well?\n2. What could be improved?\n3. Action items for next sprint",
        "discussion": "1. Problem statement\n2. Proposed solutions\n3. Technical considerations\n4. Decision and next steps",
        "one_on_one": "1. Career development\n2. Current projects\n3. Feedback\n4. Goals for next period",
        "strategy": "1. Market analysis\n2. Strategic objectives\n3. Resource allocation\n4. Timeline",
        "triage": "1. Critical bugs\n2. Bug prioritization\n3. Assignment\n4. Timeline for fixes",
        "all_hands": "1. Company updates\n2. Team achievements\n3. Upcoming initiatives\n4. Q&A"
    }
    
    return agendas.get(meeting_type, "1. Discussion topics\n2. Action items\n3. Next steps")

async def seed_database():
    """
    Main function to seed the database with sample data.
    """
    print("🌱 Starting database seeding...")
    
    # Initialize database
    db_manager.create_tables()
    print("✅ Database tables created")
    
    # Create users
    users = create_users()
    print(f"👥 Created {len(users)} users")
    
    # Generate meeting times for the next 30 days
    start_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    meeting_times = generate_meeting_times(start_date, days=30)
    print(f"📅 Generated {len(meeting_times)} meeting time slots")
    
    # Create meetings
    meetings = create_meetings(users, meeting_times)
    print(f"🤝 Created {len(meetings)} meetings")
    
    # Save to database
    with db_manager.get_session() as session:
        # Add users
        for user in users:
            session.add(user)
        
        # Add meetings
        for meeting in meetings:
            session.add(meeting)
        
        # Commit all changes
        session.commit()
        print("💾 All data saved to database")
    
    # Print summary
    print("\n📊 Database Seeding Summary:")
    print(f"   Users: {len(users)}")
    print(f"   Meetings: {len(meetings)}")
    print(f"   Date range: {start_date.strftime('%Y-%m-%d')} to {(start_date + timedelta(days=30)).strftime('%Y-%m-%d')}")
    
    print("\n✅ Database seeding completed successfully!")

if __name__ == "__main__":
    # Run the seeding process
    asyncio.run(seed_database())
