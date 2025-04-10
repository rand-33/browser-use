import asyncio
import logging
from asyncio import Event
from typing import List, Optional

logger = logging.getLogger(__name__)


class PauseHandler:
	"""
	Handler for pausing and resuming the agent.
	"""

	def __init__(
		self,
		loop: Optional[asyncio.AbstractEventLoop] = None,
		interruptible_task_patterns: Optional[List[str]] = None,
	) -> None:
		self.loop = loop or asyncio.get_event_loop()
		self.interruptible_task_patterns = interruptible_task_patterns or ['step', 'multi_act', 'get_next_action']

		# Internal state
		self._resume_event = Event()
		self._is_paused = False

	@property
	def is_paused(self) -> bool:
		"""Indicates if the handler is currently in paused state."""
		return self._is_paused

	def pause(self, interruptible_tasks: bool = True) -> None:
		"""
		Pause execution by setting the pause event.
		Note: This method is non-blocking and allows other tasks to continue.
		Agent needs to use wait_for_resume() to wait for the resume signal.
		"""
		if self._is_paused:
			logger.debug('Already paused')
			return

		self._is_paused = True

		if interruptible_tasks:
			self._cancel_interruptible_tasks()

	def resume(self) -> None:
		"""
		Resume execution by setting the resume event.
		"""
		if not self._is_paused:
			logger.debug('Not paused, skipping resume')
			return

		# Set the resume event to true
		self._resume_event.set()

	def reset(self) -> None:
		"""
		Reset the handler's internal state.
		"""
		self._is_paused = False
		self._resume_event.clear()

	async def wait_for_resume(self) -> None:
		"""
		Asynchronously wait for resume signal.
		This method is non-blocking and allows other tasks to continue.
		"""
		if not self._is_paused:
			raise Exception('Cannot wait for resume if not paused')

		try:
			# Wait for the resume event to be set to true (from resume() method)
			await self._resume_event.wait()
		finally:
			self._is_paused = False
			self._resume_event.clear()

	def _cancel_interruptible_tasks(self) -> None:
		"""Cancel current tasks that should be interruptible."""
		current_task = asyncio.current_task(self.loop)
		for task in asyncio.all_tasks(self.loop):
			if task != current_task and not task.done():
				task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
				# Cancel tasks that match certain patterns
				if any(pattern in task_name for pattern in self.interruptible_task_patterns):
					logger.debug(f'Cancelling task: {task_name}')
					task.cancel()
					# Add exception handler to silence "Task exception was never retrieved" warnings
					task.add_done_callback(lambda t: t.exception() if t.cancelled() else None)

		# Also cancel the current task if it's interruptible
		if current_task and not current_task.done():
			task_name = current_task.get_name() if hasattr(current_task, 'get_name') else str(current_task)
			if any(pattern in task_name for pattern in self.interruptible_task_patterns):
				logger.debug(f'Cancelling current task: {task_name}')
				current_task.cancel()
