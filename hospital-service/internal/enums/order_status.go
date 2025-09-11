package enums

type OrderStatus string

const (
	PENDING   OrderStatus = "PENDING"
	PAID      OrderStatus = "PAID"
	CANCELED  OrderStatus = "CANCELED"
	DELIVERED OrderStatus = "DELIVERED"
)
