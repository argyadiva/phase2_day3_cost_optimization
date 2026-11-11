# Customer Verification and Privacy

Public policy questions do not require authentication. Access to order status or customer-specific support records requires an authenticated session and an order that belongs to the authenticated customer. An order ID by itself is not proof of ownership.

The support agent must not reveal an order's items, address, status, tracking number, or support history to an unauthenticated user or to a different customer. When verification is missing, explain the limitation and ask the customer to sign in. If verification repeatedly fails, escalate to account-support without including private order details.

Never place email addresses, phone numbers, payment data, physical addresses, authentication tokens, or full customer messages in monitoring attributes. Use stable synthetic IDs and redacted summaries.

