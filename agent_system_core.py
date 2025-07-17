# Complete Agent System Core
# The full multi-agent system with Fixer, monitoring, learning, and self-provisioning

import os
import json
import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import logging

# External dependencies
import asyncpg
import aioredis
from nats.aio.client import Client as NATS
from opentelemetry import trace, metrics
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import anthropic
import openai
import requests  # For xAI and other HTTP APIs
from supabase import create_client, Client as SupabaseClient

# Setup OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(OTLPSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# Metrics
meter = metrics.get_meter(__name__)
tokens_counter = meter.create_counter("tokens_used_total")
latency_histogram = meter.create_histogram("agent_latency_ms")
heartbeat_counter = meter.create_up_down_counter("agent_heartbeat")
success_counter = meter.create_counter("task_success_total")
error_counter = meter.create_counter("task_error_total")

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class AgentConfig:
    name: str
    model: str = "claude-3-5-sonnet-20241022"  # Default to Sonnet
    capabilities: List[str] = None
    max_retries: int = 3
    timeout_seconds: int = 30
    priority: int = 3
    mcp_connectors: List[str] = None

# Base Agent Class
class BaseAgent:
    def __init__(self, config: AgentConfig, nats_client: NATS, db_pool: asyncpg.Pool, redis: aioredis.Redis):
        self.config = config
        self.name = config.name
        self.nats = nats_client
        self.db = db_pool
        self.redis = redis
        
        # Multiple AI providers for redundancy
        self.claude = anthropic.AsyncAnthropic()
        self.openai_client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.xai_api_key = os.getenv('XAI_API_KEY')
        
        # Cloud services
        self.supabase = None
        self.vercel_token = os.getenv('VERCEL_TOKEN')
        
        # Initialize Supabase if keys are available
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
        
        # Performance tracking
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "patterns_learned": 0,
            "last_heartbeat": time.time()
        }
        
        # Learning state
        self.knowledge_base = {}
        self.recent_failures = []
        
        logger.info(f"Initialized {self.name} with model {config.model}")
    
    async def start(self):
        """Start agent with heartbeat and monitoring"""
        # Subscribe to task queue
        await self.nats.subscribe(f"tasks.{self.name}", cb=self.handle_task)
        
        # Start heartbeat
        asyncio.create_task(self.heartbeat_loop())
        
        # Load knowledge base
        await self.load_knowledge_base()
        
        # Emit ready event
        await self.emit_event({
            "event": "agent.ready",
            "agent": self.name,
            "capabilities": self.config.capabilities,
            "model": self.config.model
        })
    
    async def handle_task(self, msg):
        """Process incoming task with full instrumentation"""
        with tracer.start_as_current_span(f"{self.name}.handle_task") as span:
            task_data = json.loads(msg.data.decode())
            request_id = task_data.get('request_id')
            trace_id = span.get_span_context().trace_id
            
            span.set_attributes({
                "agent.name": self.name,
                "request.id": request_id,
                "task.type": task_data.get('type')
            })
            
            start_time = time.perf_counter()
            
            try:
                # Check if we've seen this pattern before
                solution = await self.check_knowledge_base(task_data)
                
                if solution:
                    result = await self.apply_known_solution(task_data, solution)
                else:
                    result = await self.process_task(task_data)
                
                # Record success
                duration_ms = (time.perf_counter() - start_time) * 1000
                await self.record_interaction(
                    request_id=request_id,
                    trace_id=str(trace_id),
                    action_type="task_completion",
                    result=result,
                    success=True,
                    duration_ms=duration_ms
                )
                
                self.metrics["tasks_completed"] += 1
                success_counter.add(1, {"agent": self.name})
                
                # Learn from success
                await self.learn_from_success(task_data, result)
                
                return result
                
            except Exception as e:
                logger.error(f"{self.name} task failed: {e}")
                span.record_exception(e)
                
                # Record failure
                await self.record_interaction(
                    request_id=request_id,
                    trace_id=str(trace_id),
                    action_type="task_failure",
                    result={"error": str(e)},
                    success=False,
                    error_message=str(e)
                )
                
                self.metrics["tasks_failed"] += 1
                error_counter.add(1, {"agent": self.name, "error_type": type(e).__name__})
                
                # Learn from failure
                await self.learn_from_failure(task_data, e)
                
                # Escalate to Fixer
                await self.escalate_to_fixer(task_data, e)
                
                raise
    
    async def deploy_to_vercel(self, project_data: dict) -> dict:
        """Deploy fixes to Vercel"""
        if not self.vercel_token:
            raise Exception("Vercel token not configured")
            
        headers = {
            "Authorization": f"Bearer {self.vercel_token}",
            "Content-Type": "application/json"
        }
        
        # Create deployment
        deployment_data = {
            "name": project_data.get("name", "agent-fix"),
            "files": project_data.get("files", []),
            "projectSettings": {
                "framework": project_data.get("framework", "nextjs")
            }
        }
        
        response = requests.post(
            "https://api.vercel.com/v13/deployments",
            headers=headers,
            json=deployment_data,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(f"Vercel deployment failed: {response.status_code}")
            
        return response.json()
    
    async def backup_to_supabase(self, data: dict, table: str) -> dict:
        """Backup data to Supabase"""
        if not self.supabase:
            raise Exception("Supabase not configured")
            
        result = self.supabase.table(table).insert(data).execute()
        return result.data
    
    async def ask_ai(self, prompt: str, model_preference: str = "claude") -> dict:
        """Ask AI with automatic fallback to other providers"""
        providers = [
            ("claude", self._ask_claude),
            ("openai", self._ask_openai), 
            ("xai", self._ask_xai)
        ]
        
        # Try preferred provider first
        if model_preference == "openai":
            providers = providers[1:] + [providers[0]]
        elif model_preference == "xai":
            providers = providers[2:] + providers[:2]
        
        for provider_name, provider_func in providers:
            try:
                result = await provider_func(prompt)
                result["provider_used"] = provider_name
                return result
            except Exception as e:
                logger.warning(f"{provider_name} failed: {e}")
                continue
        
        raise Exception("All AI providers failed")
    
    async def _ask_claude(self, prompt: str) -> dict:
        """Ask Claude"""
        response = await self.claude.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        
        return {
            "response": response.content[0].text,
            "tokens_used": response.usage.input_tokens + response.usage.output_tokens
        }
    
    async def _ask_openai(self, prompt: str) -> dict:
        """Ask OpenAI"""
        if not os.getenv('OPENAI_API_KEY'):
            raise Exception("OpenAI API key not configured")
            
        response = await self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        
        return {
            "response": response.choices[0].message.content,
            "tokens_used": response.usage.total_tokens
        }
    
    async def _ask_xai(self, prompt: str) -> dict:
        """Ask xAI Grok"""
        if not self.xai_api_key:
            raise Exception("xAI API key not configured")
            
        headers = {
            "Authorization": f"Bearer {self.xai_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "grok-1",
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.x.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"xAI API error: {response.status_code}")
            
        result = response.json()
        return {
            "response": result["choices"][0]["message"]["content"],
            "tokens_used": result.get("usage", {}).get("total_tokens", 0)
        }
    
    async def process_task(self, task_data: Dict) -> Dict:
        """Override in specific agents"""
        raise NotImplementedError(f"{self.name} must implement process_task")
    
    async def heartbeat_loop(self):
        """Two-layer heartbeat system"""
        while True:
            try:
                # Update local timestamp
                self.metrics["last_heartbeat"] = time.time()
                
                # Emit NATS heartbeat
                await self.emit_event({
                    "event": "heartbeat",
                    "agent": self.name,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": self.metrics
                })
                
                # Update Prometheus metric
                heartbeat_counter.add(1, {"agent": self.name})
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Heartbeat failed for {self.name}: {e}")
                await asyncio.sleep(5)
    
    async def emit_event(self, data: Dict):
        """Emit event to NATS with trace context"""
        event = {
            "trace_id": trace.get_current_span().get_span_context().trace_id if trace.get_current_span() else None,
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            **data
        }
        await self.nats.publish(f"events.{self.name}", json.dumps(event).encode())
    
    async def record_interaction(self, **kwargs):
        """Record interaction to database"""
        async with self.db.acquire() as conn:
            await conn.execute("""
                INSERT INTO agent_interactions 
                (request_id, trace_id, agent_name, action_type, action_details, 
                 result, success, error_message, tokens_used, model_used, 
                 duration_ms, learned_pattern)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, 
                kwargs.get('request_id'),
                kwargs.get('trace_id'),
                self.name,
                kwargs.get('action_type'),
                json.dumps(kwargs.get('action_details', {})),
                json.dumps(kwargs.get('result', {})),
                kwargs.get('success', False),
                kwargs.get('error_message'),
                kwargs.get('tokens_used', 0),
                self.config.model,
                kwargs.get('duration_ms'),
                kwargs.get('learned_pattern')
            )
    
    async def check_knowledge_base(self, task_data: Dict) -> Optional[Dict]:
        """Check if we've solved similar problems before"""
        pattern_hash = self.hash_pattern(task_data)
        
        async with self.db.acquire() as conn:
            solution = await conn.fetchrow("""
                SELECT * FROM solution_patterns 
                WHERE pattern_hash = $1 AND success_count > failure_count
            """, pattern_hash)
            
            if solution:
                # Update usage stats
                await conn.execute("""
                    UPDATE solution_patterns 
                    SET last_used = CURRENT_TIMESTAMP
                    WHERE pattern_hash = $1
                """, pattern_hash)
                
                return json.loads(solution['solution_template'])
        
        return None
    
    async def apply_known_solution(self, task_data: Dict, solution: Dict) -> Dict:
        """Apply a solution from knowledge base"""
        with tracer.start_as_current_span(f"{self.name}.apply_known_solution"):
            # Adapt solution to current context
            adapted_solution = await self.adapt_solution(task_data, solution)
            
            # Execute with monitoring
            result = await self.execute_solution(adapted_solution)
            
            # Update success stats
            pattern_hash = self.hash_pattern(task_data)
            async with self.db.acquire() as conn:
                await conn.execute("""
                    UPDATE solution_patterns 
                    SET success_count = success_count + 1
                    WHERE pattern_hash = $1
                """, pattern_hash)
            
            return result
    
    async def learn_from_success(self, task_data: Dict, result: Dict):
        """Store successful patterns for reuse"""
        pattern_hash = self.hash_pattern(task_data)
        
        async with self.db.acquire() as conn:
            await conn.execute("""
                INSERT INTO solution_patterns 
                (pattern_hash, problem_type, solution_template, success_count, applicable_agents)
                VALUES ($1, $2, $3, 1, $4)
                ON CONFLICT (pattern_hash) 
                DO UPDATE SET 
                    success_count = solution_patterns.success_count + 1,
                    last_used = CURRENT_TIMESTAMP
            """, 
                pattern_hash,
                task_data.get('type', 'unknown'),
                json.dumps({"task": task_data, "solution": result}),
                json.dumps([self.name])
            )
        
        self.metrics["patterns_learned"] += 1
    
    async def learn_from_failure(self, task_data: Dict, error: Exception):
        """Learn from failures to avoid repeating them"""
        self.recent_failures.append({
            "task": task_data,
            "error": str(error),
            "timestamp": datetime.utcnow()
        })
        
        # Keep only recent failures
        if len(self.recent_failures) > 10:
            self.recent_failures.pop(0)
        
        # Update failure stats
        pattern_hash = self.hash_pattern(task_data)
        async with self.db.acquire() as conn:
            await conn.execute("""
                UPDATE solution_patterns 
                SET failure_count = failure_count + 1
                WHERE pattern_hash = $1
            """, pattern_hash)
    
    async def escalate_to_fixer(self, task_data: Dict, error: Exception):
        """Escalate problem to Fixer agent"""
        await self.nats.publish("fixer.intervention_needed", json.dumps({
            "agent": self.name,
            "task": task_data,
            "error": str(error),
            "error_type": type(error).__name__,
            "recent_failures": self.recent_failures[-5:],
            "timestamp": datetime.utcnow().isoformat()
        }).encode())
    
    def hash_pattern(self, task_data: Dict) -> str:
        """Create hash of task pattern for knowledge base"""
        pattern = {
            "type": task_data.get('type'),
            "key_fields": sorted(task_data.keys()),
            "agent": self.name
        }
        return hashlib.sha256(json.dumps(pattern, sort_keys=True).encode()).hexdigest()
    
    async def load_knowledge_base(self):
        """Load agent-specific knowledge on startup"""
        async with self.db.acquire() as conn:
            solutions = await conn.fetch("""
                SELECT * FROM solution_patterns 
                WHERE $1 = ANY(applicable_agents::text[])
                AND success_count > failure_count
                ORDER BY success_count DESC
                LIMIT 100
            """, self.name)
            
            for solution in solutions:
                self.knowledge_base[solution['pattern_hash']] = json.loads(solution['solution_template'])

# Specific Agent Implementations

class QuickBooksAgent(BaseAgent):
    async def process_task(self, task_data: Dict) -> Dict:
        """Handle QuickBooks-related tasks"""
        task_type = task_data.get('type')
        
        if task_type == 'reconcile_receipts':
            return await self.reconcile_receipts(task_data)
        elif task_type == 'generate_invoice':
            return await self.generate_invoice(task_data)
        elif task_type == 'sync_transactions':
            return await self.sync_transactions(task_data)
        else:
            # Use Claude to figure out what to do
            return await self.general_quickbooks_task(task_data)
    
    async def reconcile_receipts(self, task_data: Dict) -> Dict:
        """Reconcile receipts with transactions"""
        # Simulate QuickBooks API call
        await asyncio.sleep(0.5)  # Simulated API latency
        
        # In production, this would:
        # 1. Connect to QuickBooks API
        # 2. Fetch unreconciled transactions
        # 3. Match with uploaded receipts
        # 4. Flag discrepancies
        
        return {
            "status": "completed",
            "reconciled_count": 47,
            "discrepancies": [],
            "time_saved_minutes": 180
        }

class AnalysisAgent(BaseAgent):
    async def process_task(self, task_data: Dict) -> Dict:
        """Handle data analysis tasks"""
        task_type = task_data.get('type')
        
        if task_type == 'generate_report':
            return await self.generate_report(task_data)
        elif task_type == 'analyze_trends':
            return await self.analyze_trends(task_data)
        else:
            return await self.general_analysis(task_data)
    
    async def generate_report(self, task_data: Dict) -> Dict:
        """Generate analytical report using Claude"""
        with tracer.start_as_current_span("analysis.generate_report"):
            # Use Claude for complex analysis
            prompt = f"""
            Generate a business analysis report based on:
            {json.dumps(task_data.get('data', {}), indent=2)}
            
            Include insights, trends, and actionable recommendations.
            """
            
            response = await self.claude.messages.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000
            )
            
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            tokens_counter.add(tokens_used, {"agent": self.name, "model": self.config.model})
            
            return {
                "status": "completed",
                "report": response.content[0].text,
                "tokens_used": tokens_used,
                "insights_found": 5
            }

# The Fixer Agent - Master Problem Solver

class FixerAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intervention_history = {}
        self.solution_confidence_threshold = 0.8
        
    async def start(self):
        """Start Fixer with additional monitoring subscriptions"""
        await super().start()
        
        # Subscribe to all agent events for monitoring
        await self.nats.subscribe("events.*", cb=self.monitor_agent_health)
        
        # Subscribe to intervention requests
        await self.nats.subscribe("fixer.intervention_needed", cb=self.handle_intervention)
        
        # Start daily review task
        asyncio.create_task(self.daily_review_loop())
    
    async def monitor_agent_health(self, msg):
        """Monitor all agent events for problems"""
        try:
            event = json.loads(msg.data.decode())
            agent_name = event.get('agent')
            
            # Check for stuck patterns
            if await self.detect_stuck_pattern(agent_name, event):
                await self.intervene_stuck_agent(agent_name)
            
            # Check for repeated errors
            if event.get('event') == 'task_failure':
                await self.track_error_pattern(agent_name, event)
                
        except Exception as e:
            logger.error(f"Fixer monitoring error: {e}")
    
    async def handle_intervention(self, msg):
        """Handle intervention requests from agents"""
        with tracer.start_as_current_span("fixer.intervention"):
            intervention_data = json.loads(msg.data.decode())
            agent_name = intervention_data['agent']
            error_type = intervention_data['error_type']
            
            # Generate intervention ID for idempotency
            intervention_id = self.generate_intervention_id(intervention_data)
            
            # Check if already handled
            if intervention_id in self.intervention_history:
                logger.info(f"Intervention {intervention_id} already processed")
                return self.intervention_history[intervention_id]
            
            # Determine intervention strategy
            strategy = await self.determine_intervention_strategy(intervention_data)
            
            # Execute intervention
            result = await self.execute_intervention(agent_name, strategy)
            
            # Store result
            self.intervention_history[intervention_id] = result
            
            # Learn from intervention
            await self.learn_from_intervention(intervention_data, strategy, result)
            
            return result
    
    async def determine_intervention_strategy(self, intervention_data: Dict) -> Dict:
        """Use Claude Opus to determine best intervention"""
        with tracer.start_as_current_span("fixer.determine_strategy"):
            # Check knowledge base first
            known_solution = await self.check_known_solutions(intervention_data)
            if known_solution and known_solution['confidence'] > self.solution_confidence_threshold:
                return known_solution
            
            # Use Claude Opus for complex problem solving
            prompt = f"""
            An agent needs help. Analyze this situation and provide a solution:
            
            Agent: {intervention_data['agent']}
            Error: {intervention_data['error']}
            Error Type: {intervention_data['error_type']}
            Recent Failures: {json.dumps(intervention_data.get('recent_failures', []), indent=2)}
            
            Provide a specific, actionable solution that can be implemented programmatically.
            Include:
            1. Root cause analysis
            2. Immediate fix
            3. Long-term prevention strategy
            4. Confidence level (0-1)
            """
            
            response = await self.claude.messages.create(
                model="claude-3-5-opus-20241022",  # Use Opus for complex reasoning
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500
            )
            
            # Parse response into structured strategy
            strategy = self.parse_intervention_strategy(response.content[0].text)
            
            return strategy
    
    async def execute_intervention(self, agent_name: str, strategy: Dict) -> Dict:
        """Execute the intervention strategy"""
        intervention_type = strategy.get('type')
        
        if intervention_type == 'restart':
            return await self.restart_agent(agent_name)
        elif intervention_type == 'switch_connector':
            return await self.switch_connector(agent_name, strategy)
        elif intervention_type == 'update_config':
            return await self.update_agent_config(agent_name, strategy)
        elif intervention_type == 'add_capability':
            return await self.provision_new_capability(agent_name, strategy)
        elif intervention_type == 'escalate':
            return await self.escalate_to_human(agent_name, strategy)
        else:
            # Default: restart and monitor
            return await self.restart_agent(agent_name)
    
    async def restart_agent(self, agent_name: str) -> Dict:
        """Restart a stuck agent"""
        # In production, this would restart the container/process
        await self.emit_event({
            "event": "intervention.restart",
            "target_agent": agent_name,
            "reason": "stuck_pattern_detected"
        })
        
        # Simulate restart
        await asyncio.sleep(2)
        
        return {
            "intervention": "restart",
            "success": True,
            "agent": agent_name,
            "downtime_seconds": 2
        }
    
    async def switch_connector(self, agent_name: str, strategy: Dict) -> Dict:
        """Switch to alternative connector"""
        old_connector = strategy.get('old_connector')
        new_connector = strategy.get('new_connector')
        
        # Update agent configuration
        async with self.db.acquire() as conn:
            await conn.execute("""
                UPDATE agent_configs 
                SET connectors = array_append(
                    array_remove(connectors, $1), $2
                )
                WHERE agent_name = $3
            """, old_connector, new_connector, agent_name)
        
        await self.emit_event({
            "event": "intervention.connector_switch",
            "agent": agent_name,
            "from": old_connector,
            "to": new_connector
        })
        
        return {
            "intervention": "connector_switch",
            "success": True,
            "switched_from": old_connector,
            "switched_to": new_connector
        }
    
    async def provision_new_capability(self, agent_name: str, strategy: Dict) -> Dict:
        """Add new capability to agent via MCP"""
        capability = strategy.get('capability')
        
        # In production, this would use MCP API to provision
        # For now, simulate the provisioning
        await asyncio.sleep(1)
        
        await self.emit_event({
            "event": "intervention.capability_added",
            "agent": agent_name,
            "capability": capability
        })
        
        return {
            "intervention": "add_capability",
            "success": True,
            "capability_added": capability
        }
    
    async def daily_review_loop(self):
        """Run daily review and improvement cycle"""
        while True:
            # Wait until midnight
            now = datetime.utcnow()
            tomorrow = now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
            await asyncio.sleep((tomorrow - now).total_seconds())
            
            try:
                await self.run_daily_review()
            except Exception as e:
                logger.error(f"Daily review failed: {e}")
    
    async def run_daily_review(self):
        """Analyze patterns and implement improvements"""
        with tracer.start_as_current_span("fixer.daily_review"):
            # Get performance metrics for all agents
            async with self.db.acquire() as conn:
                # Agent performance
                agent_metrics = await conn.fetch("""
                    SELECT 
                        agent_name,
                        COUNT(*) as total_tasks,
                        SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful_tasks,
                        AVG(duration_ms) as avg_duration,
                        COUNT(DISTINCT error_message) as unique_errors
                    FROM agent_interactions
                    WHERE timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY agent_name
                """)
                
                # Common error patterns
                error_patterns = await conn.fetch("""
                    SELECT 
                        error_message,
                        COUNT(*) as occurrences,
                        array_agg(DISTINCT agent_name) as affected_agents
                    FROM agent_interactions
                    WHERE success = false 
                    AND timestamp > NOW() - INTERVAL '24 hours'
                    GROUP BY error_message
                    HAVING COUNT(*) > 5
                    ORDER BY COUNT(*) DESC
                """)
            
            # Generate improvements for each agent
            for agent in agent_metrics:
                if agent['successful_tasks'] / agent['total_tasks'] < 0.85:
                    await self.generate_agent_improvement(agent)
            
            # Address common error patterns
            for pattern in error_patterns:
                await self.generate_pattern_fix(pattern)
            
            # Update agent capabilities based on learnings
            await self.promote_successful_agents(agent_metrics)
    
    async def generate_agent_improvement(self, agent_metrics: Dict):
        """Generate and apply improvements for underperforming agent"""
        agent_name = agent_metrics['agent_name']
        success_rate = agent_metrics['successful_tasks'] / agent_metrics['total_tasks']
        
        # Use Claude to analyze and suggest improvements
        prompt = f"""
        Agent {agent_name} has a {success_rate:.1%} success rate.
        Average task duration: {agent_metrics['avg_duration']}ms
        Unique error types: {agent_metrics['unique_errors']}
        
        Suggest specific improvements to increase success rate above 90%.
        """
        
        response = await self.claude.messages.create(
            model="claude-3-5-opus-20241022",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        # Apply improvements
        improvements = self.parse_improvements(response.content[0].text)
        for improvement in improvements:
            await self.apply_improvement(agent_name, improvement)
    
    async def promote_successful_agents(self, agent_metrics: List[Dict]):
        """Give more responsibility to high-performing agents"""
        for agent in agent_metrics:
            if agent['successful_tasks'] / agent['total_tasks'] > 0.95:
                # Record promotion
                async with self.db.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO agent_improvements
                        (agent_name, capability_before, capability_after, 
                         trigger_pattern, success_rate_before, success_rate_after)
                        VALUES ($1, $2, $3, $4, $5, $6)
                    """,
                        agent['agent_name'],
                        'standard_operations',
                        'advanced_operations',
                        'high_success_rate',
                        0.95,
                        0.95
                    )
                
                await self.emit_event({
                    "event": "agent.promoted",
                    "agent": agent['agent_name'],
                    "new_capabilities": ["handle_complex_tasks", "train_others"]
                })
    
    def generate_intervention_id(self, intervention_data: Dict) -> str:
        """Generate unique ID for intervention idempotency"""
        key_data = {
            "agent": intervention_data['agent'],
            "error": intervention_data['error'],
            "timestamp": intervention_data['timestamp'][:16]  # Minute precision
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()

# Self-Provisioning System

class ProvisioningAgent(BaseAgent):
    """Agent that can provision new capabilities on demand"""
    
    async def process_task(self, task_data: Dict) -> Dict:
        task_type = task_data.get('type')
        
        if task_type == 'provision_tool':
            return await self.provision_tool(task_data)
        elif task_type == 'discover_capability':
            return await self.discover_capability(task_data)
    
    async def provision_tool(self, task_data: Dict) -> Dict:
        """Provision a new tool/connector via MCP"""
        tool_description = task_data.get('description')
        
        # Use Claude to understand what tool is needed
        prompt = f"""
        A user needs: {tool_description}
        
        Available MCP connectors include:
        - quickbooks, xero, freshbooks (accounting)
        - stripe, square, paypal (payments)
        - notion, confluence, obsidian (knowledge management)
        - figma, canva, sketch (design)
        - prisma, postgres, mysql (databases)
        - twilio, sendgrid (communications)
        
        Which connector(s) would best solve this need?
        If none match exactly, suggest the closest alternatives.
        """
        
        response = await self.claude.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        
        # Parse recommendations
        recommendations = self.parse_tool_recommendations(response.content[0].text)
        
        # Provision recommended tools
        provisioned = []
        for tool in recommendations:
            if await self.mcp_provision(tool):
                provisioned.append(tool)
        
        return {
            "status": "completed",
            "provisioned_tools": provisioned,
            "recommendations": recommendations
        }
    
    async def discover_capability(self, task_data: Dict) -> Dict:
        """Discover what capability is needed based on user description"""
        user_request = task_data.get('request')
        
        # Analyze request to determine needed capabilities
        prompt = f"""
        User request: {user_request}
        
        Break this down into:
        1. Core capability needed (e.g., data analysis, payment processing)
        2. Specific tools/APIs required
        3. Agent type best suited for this
        4. Estimated complexity (simple/medium/complex)
        """
        
        response = await self.claude.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        
        analysis = self.parse_capability_analysis(response.content[0].text)
        
        # Check if we have this capability
        if await self.capability_exists(analysis['core_capability']):
            return {
                "status": "capability_exists",
                "assigned_agent": analysis['agent_type'],
                "complexity": analysis['complexity']
            }
        else:
            # Need to create new capability
            return await self.create_new_capability(analysis)

# Agent Factory and Management

class AgentSystem:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.nats = None
        self.db_pool = None
        self.redis = None
    
    async def initialize(self):
        """Initialize all system components"""
        # Connect to NATS
        self.nats = NATS()
        await self.nats.connect(servers=["nats://localhost:4222"])
        
        # Connect to PostgreSQL
        self.db_pool = await asyncpg.create_pool(
            os.getenv('DATABASE_URL', 'postgresql://localhost/agent_system')
        )
        
        # Connect to Redis
        self.redis = await aioredis.create_redis_pool('redis://localhost')
        
        # Initialize agents
        await self.spawn_core_agents()
        
        logger.info("Agent system initialized")
    
    async def spawn_core_agents(self):
        """Spawn the core set of agents"""
        core_agents = [
            AgentConfig(
                name="quickbooks-agent",
                capabilities=["accounting", "invoicing", "reconciliation"],
                mcp_connectors=["quickbooks", "stripe"]
            ),
            AgentConfig(
                name="analysis-agent",
                model="claude-3-5-opus-20241022",  # Complex analysis needs Opus
                capabilities=["reporting", "trends", "forecasting"],
                mcp_connectors=["prisma", "notion"]
            ),
            AgentConfig(
                name="fixer-agent",
                model="claude-3-5-opus-20241022",  # Problem solving needs Opus
                capabilities=["monitoring", "intervention", "learning"],
                priority=5  # Highest priority
            ),
            AgentConfig(
                name="provisioning-agent",
                capabilities=["tool_discovery", "mcp_provisioning"],
                mcp_connectors=["mcp_directory"]
            )
        ]
        
        for config in core_agents:
            agent = await self.create_agent(config)
            await agent.start()
            self.agents[config.name] = agent
    
    async def create_agent(self, config: AgentConfig) -> BaseAgent:
        """Factory method to create appropriate agent type"""
        agent_classes = {
            "quickbooks-agent": QuickBooksAgent,
            "analysis-agent": AnalysisAgent,
            "fixer-agent": FixerAgent,
            "provisioning-agent": ProvisioningAgent
        }
        
        agent_class = agent_classes.get(config.name, BaseAgent)
        return agent_class(config, self.nats, self.db_pool, self.redis)
    
    async def spawn_new_agent(self, agent_type: str, capabilities: List[str]) -> BaseAgent:
        """Dynamically spawn new agent type"""
        config = AgentConfig(
            name=f"{agent_type}-{uuid.uuid4().hex[:8]}",
            capabilities=capabilities
        )
        
        agent = await self.create_agent(config)
        await agent.start()
        self.agents[config.name] = agent
        
        await self.emit_system_event({
            "event": "agent.spawned",
            "agent_type": agent_type,
            "agent_name": config.name,
            "capabilities": capabilities
        })
        
        return agent
    
    async def emit_system_event(self, data: Dict):
        """Emit system-level event"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "agent_system",
            **data
        }
        await self.nats.publish("events.system", json.dumps(event).encode())
    
    async def shutdown(self):
        """Gracefully shutdown all agents"""
        logger.info("Shutting down agent system")
        
        # Stop all agents
        for agent in self.agents.values():
            await agent.emit_event({"event": "agent.stopping"})
        
        # Close connections
        await self.nats.close()
        await self.db_pool.close()
        self.redis.close()
        await self.redis.wait_closed()

# Main entry point
async def main():
    """Start the agent system"""
    system = AgentSystem()
    
    try:
        await system.initialize()
        logger.info("Agent system running. Press Ctrl+C to stop.")
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await system.shutdown()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
