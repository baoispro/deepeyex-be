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
	DrugID    *string `json:"drug_id,omitempty"`            // nullable, thuốc có thể không có nếu chỉ là dịch vụ
	ServiceID *string `json:"service_id,omitempty"`         // nullable, dịch vụ có thể không có nếu chỉ là thuốc
	ItemName  string  `json:"item_name" binding:"required"` // gộp tên thuốc + dịch vụ, dùng hiển thị
	Price     float64 `json:"price" binding:"required"`     // giá 1 item, đã nhân theo quy cách nếu cần
	Quantity  int     `json:"quantity" binding:"required"`  // số lượng
}

type DeliveryInfo struct {
	Method  enums.DeliveryMethod `json:"method" binding:"required"`  // PICKUP, HOME_DELIVERY, EXPRESS_DELIVERY
	Address *string              `json:"address,omitempty"`          // nullable, bắt buộc nếu method là HOME_DELIVERY hoặc EXPRESS_DELIVERY
	Phone   *string              `json:"phone,omitempty"`            // nullable, số điện thoại liên hệ giao hàng
	Fullname   *string           `json:"fullname,omitempty"`            // nullable, tên người nhận
	Email   *string           	 `json:"email,omitempty"`            // nullable, email người nhận
	Notes   *string              `json:"notes,omitempty"`            // nullable, ghi chú thêm cho người giao hàng
	Fee     float64              `json:"fee"`                        // phí giao hàng, 0 nếu PICKUP
	City   *string              `json:"city,omitempty"`            // nullable, thành phố
	District   *string              `json:"district,omitempty"`            // nullable, quận
	Ward   *string              `json:"ward,omitempty"`            // nullable, phường
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
	deliveryInfo *DeliveryInfo,
) (*order.Order, error) {
	if patientID == "" || len(items) == 0 {
		return nil, errors.New("invalid order data")
	}

	// Validate delivery info
	if deliveryInfo != nil {
		if !deliveryInfo.Method.IsValid() {
			return nil, errors.New("invalid delivery method")
		}
		// Nếu là giao hàng tận nơi, bắt buộc phải có địa chỉ và số điện thoại
		if (deliveryInfo.Method == enums.HOME_DELIVERY || deliveryInfo.Method == enums.EXPRESS_DELIVERY) {
			if deliveryInfo.Address == nil || *deliveryInfo.Address == "" {
				return nil, errors.New("delivery address is required for home/express delivery")
			}
			if deliveryInfo.Phone == nil || *deliveryInfo.Phone == "" {
				return nil, errors.New("phone number is required for home/express delivery")
			}
		}
	} else {
		// Mặc định là nhận tại bệnh viện nếu không có thông tin giao hàng
		deliveryInfo = &DeliveryInfo{
			Method: enums.PICKUP,
			Fee:    0,
		}
	}

	// Bắt đầu transaction
	tx := s.repo.BeginTx()
	if tx == nil {
		return nil, errors.New("cannot start transaction")
	}

	var orderItems []order.OrderItem
	total := 0.0

	for _, it := range items {
		var drugID, serviceID string
		if it.DrugID != nil {
			drugID = *it.DrugID
		}
		if it.ServiceID != nil {
			serviceID = *it.ServiceID
		}
		// Tính tổng tiền
		total += it.Price * float64(it.Quantity)

		// Tạo order item
		orderItem := order.OrderItem{
			OrderItemID: uuid.NewString(),
			OrderID:     "",
			DrugID:      drugID,   
			ServiceID:   serviceID,
			ItemName:    it.ItemName,
			Quantity:    it.Quantity,
			Price:       it.Price,
		}
		orderItems = append(orderItems, orderItem)

		// Giảm stock nếu có DrugID
		if it.DrugID != nil {
			if err := s.drugRepo.UpdateStockAndSold(*it.DrugID, it.Quantity); err != nil {
				tx.Rollback()
				return nil, err
			}
		}
	}

	// Cộng phí giao hàng vào tổng tiền
	total += deliveryInfo.Fee

	// Tạo đơn hàng
	o := &order.Order{
		OrderID:         generateOrderID(),
		PatientID:       patientID,
		AppointmentID:   appointmentID,
		BookUserId:      bookUserID,
		CreatedAt:       time.Now(),
		Status:          status,
		TotalAmount:     total,      // tổng tiền đã tính (bao gồm phí ship)
		DeliveryMethod:  deliveryInfo.Method,
		DeliveryAddress: deliveryInfo.Address,
		DeliveryPhone:   deliveryInfo.Phone,
		DeliveryNotes:   deliveryInfo.Notes,
		DeliveryFee:     deliveryInfo.Fee,
		DeliveryCity:    deliveryInfo.City,
		DeliveryDistrict: deliveryInfo.District,
		DeliveryWard:    deliveryInfo.Ward,
		DeliveryFullname: deliveryInfo.Fullname,
		DeliveryEmail:   deliveryInfo.Email,
		OrderItems:      orderItems, // liên kết order items
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
