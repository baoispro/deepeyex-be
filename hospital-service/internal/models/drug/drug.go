package drug

import (
	"time"
)

type Drug struct {
	DrugID          string    `gorm:"column:drug_id;primaryKey;size:36" json:"drug_id"`
	Name            string    `gorm:"size:100;not null" json:"name"`
	Description     string    `gorm:"type:text" json:"description"`
	Price           float64   `gorm:"type:decimal(10,2);not null" json:"price"`
	Image           string    `json:"image" gorm:"size:255"`
	StockQuantity   int       `gorm:"not null;default:0" json:"stock_quantity"`
	DiscountPercent float64   `gorm:"type:decimal(5,2);default:0" json:"discount_percent"`
	CreatedAt       time.Time `gorm:"autoCreateTime" json:"created_at"`
	UpdatedAt       time.Time `gorm:"autoUpdateTime" json:"updated_at"`
	SoldQuantity    int       `gorm:"not null;default:0" json:"sold_quantity"`
}
