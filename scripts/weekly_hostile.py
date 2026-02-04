# scripts/weekly_hostile.py
#!/usr/bin/env python3
"""
Run weekly: Friday afternoon hostile reference session
"""
from src.training.hostile_reference import HostileReferenceInjector

injector = HostileReferenceInjector()

# Interactive session (30 min)
analysis = injector.session(interactive=True)

# Or batch mode for daily automation
# analysis = injector.session(interactive=False)
# Then later: injector.review_pending()
