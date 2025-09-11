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
	repo    *orderrepo.OrderRepo
	storage *storage.S3Client
	drugRepo *drugrepo.DrugRepo // thêm này

}

type OrderItemRequest struct {
	DrugID   string `json:"drug_id" binding:"required"`
	Quantity int    `json:"quantity" binding:"required"`
}

func NewOrderService(repo *orderrepo.OrderRepo, drugRepo *drugrepo.DrugRepo, storage *storage.S3Client) *OrderService {
	return &OrderService{
		repo:     repo,
		drugRepo: drugRepo,
		storage:  storage,
	}
}
// ---------------- CreateOrder ----------------
// func (s *OrderService) CreateOrder(patientID string, items []OrderItemRequest) (*order.Order, error) {
// 	if patientID == "" || len(items) == 0 {
// 		return nil, errors.New("invalid order data")
// 	}

// 	var orderItems []order.OrderItem
// 	total := 0.0

// 	for _, it := range items {
// 		// Lấy thông tin drug từ DB
// 		d, err := s.drugRepo.GetByID(it.DrugID)
// 		if err != nil {
// 			return nil, err
// 		}

// 		price := float64(it.Quantity) * d.Price * (1 - d.DiscountPercent/100)
// 		total += price

// 		orderItem := order.OrderItem{
// 			OrderItemID: uuid.NewString(),
// 			DrugID:      it.DrugID,
// 			Quantity:    it.Quantity,
// 			Price:       price,
// 		}
// 		orderItems = append(orderItems, orderItem)

		
// 	}

// 	o := &order.Order{
// 		OrderID:     generateOrderID(),
// 		PatientID:   patientID,
// 		CreatedAt:   time.Now(),
// 		Status:      enums.PENDING,
// 		TotalAmount: total,
// 		OrderItems:  orderItems,
// 	}

// 	if err := s.repo.Create(o); err != nil {
// 		return nil, err
// 	}

// 	return o, nil
// }

func (s *OrderService) CreateOrder(patientID string, items []OrderItemRequest) (*order.Order, error) {
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
			DrugID:      it.DrugID,
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
		OrderID:     generateOrderID(),
		PatientID:   patientID,
		CreatedAt:   time.Now(),
		Status:      enums.PENDING,
		TotalAmount: total,
		OrderItems:  orderItems,
	}

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

// ---------------- DeleteOrder ----------------
func (s *OrderService) DeleteOrder(id string) error {
	return s.repo.Delete(id)
}

// ---------------- Helper ----------------
func generateOrderID() string {
	return uuid.NewString()
}

// ---------------- Update Order Detail ----------------
func (s *OrderService) UpdateOrderDetail(id string, updated *order.Order) error {
	o, err := s.repo.GetByID(id)
	if err != nil {
		return err
	}

	if len(updated.OrderItems) > 0 {
		total := 0.0
		for i := range updated.OrderItems {
			if updated.OrderItems[i].OrderItemID == "" {
				updated.OrderItems[i].OrderItemID = uuid.NewString()
			}
			updated.OrderItems[i].Price = updated.OrderItems[i].Price * float64(updated.OrderItems[i].Quantity)
			total += updated.OrderItems[i].Price
		}
		o.OrderItems = updated.OrderItems
		o.TotalAmount = total
	}

	if updated.Status != "" {
		o.Status = updated.Status
	}

	return s.repo.Update(o)
}


// ---------------- GetOrdersByPatientID ----------------
func (s *OrderService) GetOrdersByPatientID(patientID string) ([]order.Order, error) {
	return s.repo.FindByPatientID(patientID)
}
