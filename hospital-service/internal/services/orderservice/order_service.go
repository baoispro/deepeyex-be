package orderservice

import (
	"errors"
	"hospital-service/internal/enums"
	"hospital-service/internal/models/order"
	"hospital-service/internal/repositories/drugrepo"
	"hospital-service/internal/repositories/orderrepo"
	"hospital-service/internal/storage"
	"time"

	"github.com/google/uuid"
)

type OrderService struct {
	repo     *orderrepo.OrderRepo
	storage  *storage.S3Client
	drugRepo *drugrepo.DrugRepo // thêm này

}

type OrderItemRequest struct {
	DrugID   string `json:"drug_id"`
	Quantity int    `json:"quantity" binding:"required"`
	Service  string `json:"service"`
}

func NewOrderService(repo *orderrepo.OrderRepo, drugRepo *drugrepo.DrugRepo, storage *storage.S3Client) *OrderService {
	return &OrderService{
		repo:     repo,
		drugRepo: drugRepo,
		storage:  storage,
	}
}

func (s *OrderService) CreateOrder(
	patientID string,
	appointmentID string,
	bookUserID string,
	status enums.OrderStatus,
	items []OrderItemRequest,
) (*order.Order, error) {
	if patientID == "" || len(items) == 0 {
		return nil, errors.New("invalid order data")
	}

	// Bắt đầu transaction
	tx := s.repo.BeginTx()
	if tx == nil {
		return nil, errors.New("cannot start transaction")
	}

	var orderItems []order.OrderItem
	total := 0.0

	for _, it := range items {
		// Lấy thông tin thuốc
		d, err := s.drugRepo.GetByID(it.DrugID)
		if err != nil {
			tx.Rollback()
			return nil, err
		}

		// Tính giá sau khi giảm giá
		price := float64(it.Quantity) * d.Price * (1 - d.DiscountPercent/100)
		total += price

		// Tạo order item
		orderItem := order.OrderItem{
			OrderItemID: uuid.NewString(),
			OrderID:     "", // sẽ được gán sau khi tạo Order
			DrugID:      it.DrugID,
			DrugName:    d.Name,  // populate tên thuốc
			Service:     it.Service,
			Quantity:    it.Quantity,
			Price:       price,
		}
		orderItems = append(orderItems, orderItem)

		// Giảm stock và tăng sold quantity
		if err := s.drugRepo.UpdateStockAndSold(it.DrugID, it.Quantity); err != nil {
			tx.Rollback()
			return nil, err
		}
	}

	// Tạo đơn hàng
	o := &order.Order{
		OrderID:       generateOrderID(),
		PatientID:     patientID,
		AppointmentID: appointmentID,
		BookUserId:    bookUserID,
		CreatedAt:     time.Now(),
		Status:        status,
		TotalAmount:   total,
		OrderItems:    orderItems,
	}

	// Lưu Order (bao gồm OrderItems)
	if err := tx.Create(o).Error; err != nil {
		tx.Rollback()
		return nil, err
	}

	// Commit transaction
	if err := tx.Commit().Error; err != nil {
		tx.Rollback()
		return nil, err
	}

	return o, nil
}

// ---------------- GetOrder ----------------
func (s *OrderService) GetOrder(id string) (*order.Order, error) {
	return s.repo.GetByID(id)
}

// ---------------- ListOrders ----------------
func (s *OrderService) ListOrders() ([]order.Order, error) {
	return s.repo.ListAll()
}

// ---------------- UpdateOrderStatus ----------------
func (s *OrderService) UpdateOrderStatus(id string, status enums.OrderStatus) error {
	o, err := s.repo.GetByID(id)
	if err != nil {
		return err
	}
	o.Status = status
	return s.repo.Update(o)
}

// ---------------- UpdateOrderAppointment ----------------
func (s *OrderService) UpdateOrderAppointment(id string, appointmentID string) error {
	o, err := s.repo.GetByID(id)
	if err != nil {
		return err
	}

	// cập nhật appointment_id
	o.AppointmentID = appointmentID

	return s.repo.Update(o)
}

// ---------------- DeleteOrder ----------------
func (s *OrderService) DeleteOrder(id string) error {
	return s.repo.Delete(id)
}

// ---------------- Helper ----------------
func generateOrderID() string {
	return uuid.NewString()
}

// ---------------- GetOrdersByPatientID ----------------
func (s *OrderService) GetOrdersByPatientID(patientID string) ([]order.Order, error) {
	return s.repo.FindByPatientID(patientID)
}
