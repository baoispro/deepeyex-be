package enums

type AppointmentStatus string

const (
	Pending   AppointmentStatus = "PENDING"
	Confirmed AppointmentStatus = "CONFIRMED"
	Completed AppointmentStatus = "COMPLETED"
	Canceled  AppointmentStatus = "CANCELED"
)
