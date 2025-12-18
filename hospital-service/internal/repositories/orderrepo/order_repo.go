package orderrepo

import (
	"hospital-service/internal/models/order"
	"strings"
	"time"

	"gorm.io/gorm"
)

type OrderRepo struct {
	db *gorm.DB
}

func NewOrderRepo(db *gorm.DB) *OrderRepo {
	return &OrderRepo{db: db}
}

// ---------------- Create ----------------
func (r *OrderRepo) Create(o *order.Order) error {
	return r.db.Create(o).Error
}

// ---------------- GetByID ----------------
func (r *OrderRepo) GetByID(id string) (*order.Order, error) {
	var o order.Order
	if err := r.db.Preload("OrderItems.Drug").Preload("Patient").First(&o, "order_id = ?", id).Error; err != nil {
		return nil, err
	}
	return &o, nil
}

// -------------- GetByPatientID --------------
// Chỉ lấy orders không có appointment_id VÀ có ít nhất 1 order_item chứa drug_id
func (r *OrderRepo) FindByPatientID(patientID string) ([]order.Order, error) {
	var orders []order.Order
	
	// Join với order_items để chỉ lấy orders CÓ order_items có drug_id
	// Distinct để tránh duplicate orders khi join
	err := r.db.
		Joins("JOIN order_items ON order_items.order_id = orders.order_id").
		Where("orders.patient_id = ?", patientID).
		Distinct().
		Preload("OrderItems").
		Preload("OrderItems.Drug").
		Preload("OrderItems.Service").
		Preload("Patient").
		Find(&orders).Error
	
	if err != nil {
		return nil, err
	}
	
	return orders, nil
}

// ---------------- ListAll ----------------
func (r *OrderRepo) ListAll() ([]order.Order, error) {
	var orders []order.Order
	if err := r.db.Preload("OrderItems.Drug").Preload("Patient").Find(&orders).Error; err != nil {
		return nil, err
	}
	return orders, nil
}

// FindWithFilters tìm orders với filter động
func (r *OrderRepo) FindWithFilters(status, orderDate string) ([]order.Order, error) {
	var orders []order.Order
	query := r.db.Preload("OrderItems.Drug").Preload("Patient")

	// Filter theo status (exact match)
	if status != "" {
		query = query.Where("status = ?", strings.ToUpper(status))
	}

	// Filter theo order date (format: YYYY-MM-DD)
	if orderDate != "" {
		// Parse date và filter theo ngày (bỏ qua giờ)
		if parsedDate, err := time.Parse("2006-01-02", orderDate); err == nil {
			// Filter từ 00:00:00 đến 23:59:59 của ngày đó
			startOfDay := parsedDate
			endOfDay := parsedDate.Add(24 * time.Hour).Add(-1 * time.Second)
			query = query.Where("created_at >= ? AND created_at <= ?", startOfDay, endOfDay)
		}
	}

	if err := query.Find(&orders).Error; err != nil {
		return nil, err
	}
	return orders, nil
}

// ---------------- Update ----------------
func (r *OrderRepo) Update(o *order.Order) error {
	return r.db.Save(o).Error
}

// ---------------- Delete ----------------
func (r *OrderRepo) Delete(id string) error {
	return r.db.Delete(&order.Order{}, "order_id = ?", id).Error
}

func (r *OrderRepo) BeginTx() *gorm.DB {
	return r.db.Begin()
}
