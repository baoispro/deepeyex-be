package orderrepo

import (
	"hospital-service/internal/models/order"

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
		Where("orders.appointment_id IS NULL OR orders.appointment_id = '' OR orders.appointment_id = 'EMPTY'").
		Where("order_items.drug_id IS NOT NULL AND order_items.drug_id != ''").
		Distinct().
		Preload("OrderItems", "drug_id IS NOT NULL AND drug_id != ''").
		Preload("OrderItems.Drug").
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
