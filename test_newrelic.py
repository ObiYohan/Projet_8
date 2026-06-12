import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

application = newrelic.agent.register_application(timeout=10)
print(f"Application registered: {application}")