# Multi-Agent AI System

A self-improving multi-agent system that learns from user requests and automatically provisions new capabilities.

## What It Does

Users describe problems in plain English through a web portal. The system automatically:
- Assigns the right AI agents to solve the problem
- Learns from successes and failures  
- Self-heals when agents get stuck
- Provisions new tools/capabilities as needed

## Key Components

### 1. Request Portal (`portal_and_database.py`)
- Clean web interface for problem submission
- FastAPI backend with PostgreSQL
- Automatic agent assignment based on problem type
- Real-time progress tracking

### 2. Agent System Core (`agent_system_core.py`)
- Multi-agent orchestration with NATS messaging
- **Fixer Agent** - Monitors and fixes stuck agents
- **Provisioning Agent** - Adds new capabilities on demand
- **Specialized Agents** - QuickBooks, Analysis, etc.
- Full observability with OpenTelemetry

### 3. Claude CLI (`claude_client.py`)
- Simple command-line interface to Claude
- Works from any directory
- Interactive and one-shot modes

## Architecture

```
User Request → Portal → Agent Assignment → Task Execution → Learning
                ↓              ↓              ↓           ↓
            Database ←→ Message Queue ←→ Monitoring ←→ Auto-Improvement
```

## Quick Start

### 1. One-Command Setup (Recommended)
```bash
# Windows
start.bat

# Linux/Mac
./start.sh
```
This will:
- Check Docker is running
- Set up environment variables
- Start all services (database, agents, monitoring)
- Give you URLs to access everything

### 2. Manual Setup
```bash
# 1. Copy environment template
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY

# 2. Start all services
docker-compose up -d

# 3. Check everything is running
docker-compose ps
```

### 3. Submit your first request
Open http://localhost:8000 and describe your problem:
- "I spend 3 hours every Monday reconciling receipts"
- "Customers don't know their order status"
- "Creating monthly reports takes forever"

### 4. Monitor the system
- **Portal**: http://localhost:8000 - Submit requests and track progress
- **Grafana**: http://localhost:3000 - Performance dashboards (admin/admin)
- **Prometheus**: http://localhost:9090 - Raw metrics
- **NATS**: http://localhost:8222 - Message queue monitoring

## Key Features

### Self-Healing
- Agents monitor each other for stuck patterns
- Fixer automatically restarts/reconfigures failed agents
- Learns from interventions to prevent future problems

### Auto-Learning
- Every successful pattern is stored for reuse
- Failed attempts trigger automatic improvements
- Daily reviews optimize agent performance

### Auto-Provisioning
- System detects when new capabilities are needed
- Automatically provisions MCP connectors
- Spawns new specialized agents on demand

### Full Observability
- Every action is traced and logged
- Prometheus metrics for performance monitoring
- Detailed audit trail for debugging

## Database Schema

See `portal_and_database.py` for the complete schema including:
- Request tracking
- Agent interactions and learning
- Solution pattern library
- Performance metrics
- Feedback loops

## Agent Types

- **QuickBooks Agent** - Accounting, invoicing, reconciliation
- **Analysis Agent** - Reports, trends, forecasting (uses Claude Opus)
- **Fixer Agent** - System monitoring and problem resolution
- **Provisioning Agent** - Dynamic capability addition
- **Communication Agent** - Email, notifications
- **UI Agent** - Website and interface tasks

## Production Deployment

The system is designed for:
- Kubernetes deployment with auto-scaling
- NATS for reliable message delivery
- PostgreSQL for persistent storage
- Redis for caching and sessions
- OpenTelemetry for observability

## Development

### Adding New Agent Types
1. Inherit from `BaseAgent`
2. Implement `process_task()`
3. Add to agent factory in `AgentSystem.create_agent()`

### Adding New Capabilities
The Fixer can automatically add capabilities via MCP connectors.
Or manually add to an agent's `capabilities` list.

## Requirements

- Python 3.9+
- PostgreSQL 12+
- NATS 2.9+
- Redis 6+
- Claude API access

## License

MIT License - See LICENSE file for details.
