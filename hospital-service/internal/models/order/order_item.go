package order

import "hospital-service/internal/models/drug"

type OrderItem struct {
	OrderItemID string     `gorm:"column:order_item_id;primaryKey;size:36" json:"order_item_id"`
	OrderID     string     `gorm:"column:order_id;size:36;not null" json:"order_id"`
	DrugID      string     `gorm:"column:drug_id;size:36;not null" json:"drug_id"`
	Quantity    int        `gorm:"not null;default:1" json:"quantity"`
	Price       float64    `gorm:"type:decimal(10,2);not null" json:"price"`
	Drug        drug.Drug `gorm:"foreignKey:DrugID;references:DrugID" json:"drug"` // <-- thêm references
}