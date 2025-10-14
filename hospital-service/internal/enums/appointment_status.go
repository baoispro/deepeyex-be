package enums

type AppointmentStatus string

const (
	Pending   AppointmentStatus = "PENDING"
	Confirmed AppointmentStatus = "CONFIRMED"
	Completed AppointmentStatus = "COMPLETED"
	Canceled  AppointmentStatus = "CANCELED"

	PendingOnline   AppointmentStatus = "PENDING_ONLINE"
	ConfirmedOnline AppointmentStatus = "CONFIRMED_ONLINE"
	CompletedOnline AppointmentStatus = "COMPLETED_ONLINE"
)
