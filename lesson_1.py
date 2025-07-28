from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
import hashlib
import secrets
import re
from decimal import Decimal


# Enums для типов и статусов
class UserRole(Enum):
    """Роли пользователей в системе"""
    USER = "user"
    ADMIN = "admin"


class TransactionType(Enum):
    """Типы транзакций"""
    DEPOSIT = "deposit"  # Пополнение баланса
    WITHDRAWAL = "withdrawal"  # Вывод средств
    PREDICTION_FEE = "prediction_fee"  # Списание за предсказание
    REFUND = "refund"  # Возврат средств


class TransactionStatus(Enum):
    """Статусы транзакций"""
    PENDING = "pending"  # Ожидает модерации (для пополнений)
    COMPLETED = "completed"  # Завершена
    FAILED = "failed"  # Отклонена
    CANCELLED = "cancelled"  # Отменена


class PredictionStatus(Enum):
    """Статусы предсказаний"""
    PENDING = "pending"  # В очереди
    PROCESSING = "processing"  # Обрабатывается воркером
    COMPLETED = "completed"  # Успешно завершено
    FAILED = "failed"  # Ошибка при обработке
    VALIDATION_ERROR = "validation_error"  # Ошибка валидации данных
    PARTIALLY_COMPLETED = "partially_completed"  # Частично выполнено (для batch)


# Базовый класс
@dataclass
class BaseEntity(ABC):
    """Базовый класс для всех сущностей"""
    id: Optional[int] = field(default=None)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def update_timestamp(self) -> None:
        """Обновляет временную метку изменения"""
        self.updated_at = datetime.now()


# Пользователи
@dataclass
class User(BaseEntity):
    """
    Базовый класс пользователя системы
    
    Инкапсулирует данные пользователя и основные операции с балансом
    """
    email: str
    _password_hash: str = field(repr=False)  # Приватное поле
    balance: Decimal = field(default=Decimal('0.00'))
    role: UserRole = field(default=UserRole.USER)
    is_active: bool = field(default=True)
    api_key: Optional[str] = field(default=None)
    telegram_id: Optional[int] = field(default=None)  # Для Telegram bot
    
    def __post_init__(self) -> None:
        super().__init__()
        self._validate_email()
        self._validate_balance()
        if not self.api_key:
            self.api_key = self._generate_api_key()
    
    def _validate_email(self) -> None:
        """Валидация email"""
        pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        if not pattern.match(self.email):
            raise ValueError(f"Invalid email format: {self.email}")
    
    def _validate_balance(self) -> None:
        """Валидация баланса"""
        if self.balance < 0:
            raise ValueError("Balance cannot be negative")
    
    def _generate_api_key(self) -> str:
        """Генерация уникального API ключа"""
        return secrets.token_urlsafe(32)
    
    @property
    def password_hash(self) -> str:
        """Геттер для хэша пароля"""
        return self._password_hash
    
    def set_password(self, password: str) -> None:
        """Устанавливает новый пароль (хэшированный)"""
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")
        self._password_hash = hashlib.sha256(password.encode()).hexdigest()
        self.update_timestamp()
    
    def verify_password(self, password: str) -> bool:
        """Проверяет пароль"""
        return hashlib.sha256(password.encode()).hexdigest() == self._password_hash
    
    def can_afford(self, amount: Decimal) -> bool:
        """Проверяет достаточность средств"""
        return self.balance >= amount
    
    def withdraw(self, amount: Decimal) -> None:
        """Списывает средства"""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        if not self.can_afford(amount):
            raise ValueError(f"Insufficient balance. Required: {amount}, Available: {self.balance}")
        self.balance -= amount
        self.update_timestamp()
    
    def deposit(self, amount: Decimal) -> None:
        """Пополняет баланс"""
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self.balance += amount
        self.update_timestamp()


@dataclass
class Admin(User):
    """Администратор с расширенными правами"""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.role = UserRole.ADMIN
    
    def approve_transaction(self, transaction: 'Transaction') -> None:
        """Одобряет транзакцию"""
        if transaction.type != TransactionType.DEPOSIT:
            raise ValueError("Can only approve deposit transactions")
        if transaction.status != TransactionStatus.PENDING:
            raise ValueError("Can only approve pending transactions")
        transaction.approve()
    
    def reject_transaction(self, transaction: 'Transaction', reason: str) -> None:
        """Отклоняет транзакцию"""
        if transaction.type != TransactionType.DEPOSIT:
            raise ValueError("Can only reject deposit transactions")
        if transaction.status != TransactionStatus.PENDING:
            raise ValueError("Can only reject pending transactions")
        transaction.reject(reason)


# ML Модели
@dataclass
class MLModel(BaseEntity):
    """
    ML модель доступная в системе
    """
    name: str
    description: str
    version: str
    credit_cost: Decimal
    is_active: bool = field(default=True)
    
    # Конфигурация модели
    model_type: str = field(default="")  # classification, regression, etc.
    input_fields: List[Dict[str, str]] = field(default_factory=list)  # [{"name": "text", "type": "string", "required": true}]
    output_fields: List[Dict[str, str]] = field(default_factory=list)
    max_batch_size: int = field(default=1)
    timeout_seconds: int = field(default=300)
    
    def __post_init__(self) -> None:
        super().__init__()
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Валидация конфигурации модели"""
        if self.credit_cost <= 0:
            raise ValueError("Credit cost must be positive")
        if not self.name:
            raise ValueError("Model name is required")
        if self.max_batch_size < 1:
            raise ValueError("Max batch size must be at least 1")
    
    def validate_input_data(self, data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Валидирует входные данные
        Returns: (is_valid, error_messages)
        """
        errors = []
        
        for field in self.input_fields:
            field_name = field.get('name')
            field_type = field.get('type')
            is_required = field.get('required', False)
            
            if is_required and field_name not in data:
                errors.append(f"Required field '{field_name}' is missing")
                continue
            
            if field_name in data:
                value = data[field_name]
                # Простая проверка типов
                if field_type == 'string' and not isinstance(value, str):
                    errors.append(f"Field '{field_name}' must be string")
                elif field_type == 'number' and not isinstance(value, (int, float)):
                    errors.append(f"Field '{field_name}' must be number")
                elif field_type == 'array' and not isinstance(value, list):
                    errors.append(f"Field '{field_name}' must be array")
        
        return len(errors) == 0, errors


# Транзакции
@dataclass
class Transaction(BaseEntity):
    """
    Финансовая транзакция в системе
    """
    user_id: int
    amount: Decimal
    type: TransactionType
    status: TransactionStatus = field(default=TransactionStatus.PENDING)
    description: str = field(default="")
    admin_id: Optional[int] = field(default=None)  # ID админа для модерации
    rejection_reason: Optional[str] = field(default=None)
    
    # Связи
    related_prediction_id: Optional[int] = field(default=None)
    
    def __post_init__(self) -> None:
        super().__init__()
        if self.amount <= 0:
            raise ValueError("Transaction amount must be positive")
        
        # Автоматически завершаем некоторые типы транзакций
        if self.type in [TransactionType.PREDICTION_FEE, TransactionType.WITHDRAWAL]:
            self.status = TransactionStatus.COMPLETED
    
    def approve(self) -> None:
        """Одобряет транзакцию"""
        self.status = TransactionStatus.COMPLETED
        self.update_timestamp()
    
    def reject(self, reason: str) -> None:
        """Отклоняет транзакцию"""
        self.status = TransactionStatus.FAILED
        self.rejection_reason = reason
        self.update_timestamp()
    
    def cancel(self) -> None:
        """Отменяет транзакцию"""
        if self.status == TransactionStatus.COMPLETED:
            raise ValueError("Cannot cancel completed transaction")
        self.status = TransactionStatus.CANCELLED
        self.update_timestamp()


# Задачи и предсказания
@dataclass
class PredictionTask(BaseEntity):
    """
    Задача для ML воркера (для RabbitMQ)
    """
    user_id: int
    model_id: int
    input_data: Dict[str, Any]
    priority: int = field(default=0)  # Приоритет обработки
    
    # Обработка
    status: PredictionStatus = field(default=PredictionStatus.PENDING)
    worker_id: Optional[str] = field(default=None)
    retry_count: int = field(default=0)
    max_retries: int = field(default=3)
    
    def assign_worker(self, worker_id: str) -> None:
        """Назначает воркера для обработки"""
        self.worker_id = worker_id
        self.status = PredictionStatus.PROCESSING
        self.update_timestamp()
    
    def increment_retry(self) -> bool:
        """
        Увеличивает счетчик попыток
        Returns: True если еще можно повторить
        """
        self.retry_count += 1
        return self.retry_count < self.max_retries
    
    def to_message(self) -> Dict[str, Any]:
        """Преобразует в сообщение для очереди"""
        return {
            'task_id': self.id,
            'model_id': self.model_id,
            'user_id': self.user_id,
            'input_data': self.input_data,
            'priority': self.priority,
            'retry_count': self.retry_count
        }


@dataclass
class Prediction(BaseEntity):
    """
    Результат ML предсказания
    """
    task_id: int
    user_id: int
    model_id: int
    
    # Данные
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = field(default=None)
    valid_data: Optional[Dict[str, Any]] = field(default=None)  # Валидные данные из batch
    invalid_data: Optional[Dict[str, Any]] = field(default=None)  # Невалидные данные
    
    # Статус и метрики
    status: PredictionStatus = field(default=PredictionStatus.PENDING)
    credits_charged: Decimal = field(default=Decimal('0'))
    processing_time_ms: Optional[int] = field(default=None)
    error_message: Optional[str] = field(default=None)
    
    # Связи
    transaction_id: Optional[int] = field(default=None)
    
    def set_completed(self, output: Dict[str, Any], processing_time: int) -> None:
        """Успешное завершение"""
        self.output_data = output
        self.processing_time_ms = processing_time
        self.status = PredictionStatus.COMPLETED
        self.update_timestamp()
    
    def set_partial(self, valid: Dict[str, Any], invalid: Dict[str, Any],
                    output: Dict[str, Any], processing_time: int) -> None:
        """Частичное выполнение (для batch)"""
        self.valid_data = valid
        self.invalid_data = invalid
        self.output_data = output
        self.processing_time_ms = processing_time
        self.status = PredictionStatus.PARTIALLY_COMPLETED
        self.update_timestamp()
    
    def set_failed(self, error: str) -> None:
        """Ошибка выполнения"""
        self.error_message = error
        self.status = PredictionStatus.FAILED
        self.update_timestamp()
    
    def set_validation_error(self, errors: List[str]) -> None:
        """Ошибка валидации"""
        self.error_message = "; ".join(errors)
        self.status = PredictionStatus.VALIDATION_ERROR
        self.update_timestamp()


# Интерфейсы для различных способов взаимодействия
@dataclass
class AuthSession(BaseEntity):
    """Сессия авторизации"""
    user_id: int
    token: str
    interface_type: str  # 'web', 'api', 'telegram'
    expires_at: datetime
    is_active: bool = field(default=True)
    
    def is_expired(self) -> bool:
        """Проверяет истекла ли сессия"""
        return datetime.now() > self.expires_at
    
    def revoke(self) -> None:
        """Отзывает сессию"""
        self.is_active = False
        self.update_timestamp()


# Сервисные классы для бизнес-логики
class UserService:
    """Сервис для работы с пользователями"""
    def __init__(self):
        self._users: Dict[int, User] = {}
        self._next_id = 1
    
    def create_user(self, email: str, password: str) -> User:
        """Создает нового пользователя"""
        user = User(
            id=self._next_id,
            email=email,
            _password_hash=""
        )
        user.set_password(password)
        self._users[user.id] = user
        self._next_id += 1
        return user
    
    def get_by_email(self, email: str) -> Optional[User]:
        """Поиск по email"""
        for user in self._users.values():
            if user.email == email:
                return user
        return None
    
    def get_by_api_key(self, api_key: str) -> Optional[User]:
        """Поиск по API ключу"""
        for user in self._users.values():
            if user.api_key == api_key:
                return user
        return None
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Аутентификация пользователя"""
        user = self.get_by_email(email)
        if user and user.verify_password(password):
            return user
        return None


class MLService:
    """Сервис для работы с ML"""
    def __init__(self):
        self._models: Dict[int, MLModel] = {}
        self._tasks: Dict[int, PredictionTask] = {}
        self._predictions: Dict[int, Prediction] = {}
        self._next_id = 1
    
    def create_prediction_request(self, user: User, model: MLModel,
                                 input_data: Dict[str, Any]) -> PredictionTask:
        """Создает запрос на предсказание"""
        # Проверяем баланс
        if not user.can_afford(model.credit_cost):
            raise ValueError(f"Insufficient balance. Need {model.credit_cost}, have {user.balance}")
        
        # Валидируем данные
        is_valid, errors = model.validate_input_data(input_data)
        if not is_valid:
            raise ValueError(f"Validation errors: {'; '.join(errors)}")
        
        # Создаем задачу
        task = PredictionTask(
            id=self._next_id,
            user_id=user.id,
            model_id=model.id,
            input_data=input_data
        )
        self._tasks[task.id] = task
        
        # Создаем предсказание
        prediction = Prediction(
            id=self._next_id,
            task_id=task.id,
            user_id=user.id,
            model_id=model.id,
            input_data=input_data,
            credits_charged=model.credit_cost
        )
        self._predictions[prediction.id] = prediction
        
        self._next_id += 1
        return task
    
    def complete_prediction(self, task_id: int, result: Dict[str, Any],
                          processing_time: int) -> Prediction:
        """Завершает предсказание"""
        task = self._tasks.get(task_id)
        if not task:
            raise ValueError("Task not found")
        
        prediction = next((p for p in self._predictions.values()
                          if p.task_id == task_id), None)
        if not prediction:
            raise ValueError("Prediction not found")
        
        prediction.set_completed(result, processing_time)
        task.status = PredictionStatus.COMPLETED
        return prediction
    
    def get_user_predictions(self, user_id: int) -> List[Prediction]:
        """Получает историю предсказаний пользователя"""
        return [p for p in self._predictions.values() if p.user_id == user_id]


class TransactionService:
    """Сервис для работы с транзакциями"""
    def __init__(self):
        self._transactions: Dict[int, Transaction] = {}
        self._next_id = 1
    
    def create_deposit_request(self, user: User, amount: Decimal) -> Transaction:
        """Создает запрос на пополнение (требует модерации)"""
        transaction = Transaction(
            id=self._next_id,
            user_id=user.id,
            amount=amount,
            type=TransactionType.DEPOSIT,
            status=TransactionStatus.PENDING,
            description=f"Deposit request: {amount} credits"
        )
        self._transactions[transaction.id] = transaction
        self._next_id += 1
        return transaction
    
    def charge_for_prediction(self, user: User, prediction: Prediction,
                            model: MLModel) -> Transaction:
        """Списывает средства за предсказание"""
        transaction = Transaction(
            id=self._next_id,
            user_id=user.id,
            amount=model.credit_cost,
            type=TransactionType.PREDICTION_FEE,
            status=TransactionStatus.COMPLETED,
            description=f"Prediction with {model.name} v{model.version}",
            related_prediction_id=prediction.id
        )
        
        # Списываем средства
        user.withdraw(model.credit_cost)
        
        self._transactions[transaction.id] = transaction
        prediction.transaction_id = transaction.id
        self._next_id += 1
        return transaction
    
    def get_user_transactions(self, user_id: int) -> List[Transaction]:
        """История транзакций пользователя"""
        return [t for t in self._transactions.values() if t.user_id == user_id]
    
    def get_pending_deposits(self) -> List[Transaction]:
        """Получает депозиты ожидающие модерации (для админов)"""
        return [t for t in self._transactions.values()
                if t.type == TransactionType.DEPOSIT and t.status == TransactionStatus.PENDING]


# Пример использования
def demo():
    # Инициализация сервисов
    user_service = UserService()
    ml_service = MLService()
    transaction_service = TransactionService()
    
    # Создаем пользователей
    user = user_service.create_user("user@example.com", "password123")
    admin_user = user_service.create_user("admin@example.com", "adminpass123")
    admin = Admin(
        id=admin_user.id,
        email=admin_user.email,
        _password_hash=admin_user.password_hash,
        api_key=admin_user.api_key
    )
    
    # Создаем ML модель
    model = MLModel(
        id=1,
        name="Sentiment Analysis",
        description="Analyzes text sentiment",
        version="2.0.1",
        credit_cost=Decimal("2.50"),
        model_type="classification",
        input_fields=[
            {"name": "text", "type": "string", "required": True},
            {"name": "language", "type": "string", "required": False}
        ],
        output_fields=[
            {"name": "sentiment", "type": "string"},
            {"name": "confidence", "type": "number"}
        ]
    )
    
    # Пользователь запрашивает пополнение
    deposit_request = transaction_service.create_deposit_request(user, Decimal("100.00"))
    print(f"Deposit request created: {deposit_request.id}")
    
    # Админ одобряет пополнение
    admin.approve_transaction(deposit_request)
    user.deposit(deposit_request.amount)
    print(f"User balance after deposit: {user.balance}")
    
    # Создаем задачу предсказания
    try:
        task = ml_service.create_prediction_request(
            user=user,
            model=model,
            input_data={"text": "This product is amazing!", "language": "en"}
        )
        print(f"Prediction task created: {task.id}")
        
        # Имитируем обработку воркером
        task.assign_worker("worker-001")
        
        # Завершаем предсказание
        prediction = ml_service.complete_prediction(
            task_id=task.id,
            result={"sentiment": "positive", "confidence": 0.98},
            processing_time=125
        )
        
        # Списываем средства
        transaction = transaction_service.charge_for_prediction(user, prediction, model)
        print(f"Prediction completed. Credits charged: {transaction.amount}")
        print(f"User balance after prediction: {user.balance}")
        
    except ValueError as e:
        print(f"Error: {e}")
    
    # Проверяем историю
    user_transactions = transaction_service.get_user_transactions(user.id)
    user_predictions = ml_service.get_user_predictions(user.id)
    
    print(f"\nUser transactions: {len(user_transactions)}")
    print(f"User predictions: {len(user_predictions)}")


if __name__ == "__main__":
    demo()
