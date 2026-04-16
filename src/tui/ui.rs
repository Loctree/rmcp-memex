//! TUI Rendering Module
//!
//! Renders the wizard UI using ratatui widgets.

use crate::tui::app::{App, WizardStep};
use crate::tui::detection::ProviderStatus;
use crate::tui::health::CheckStatus;
use crate::tui::indexer::{DataSetupOption, DataSetupSubStep, ImportMode, IndexTelemetrySnapshot};
use crate::tui::monitor::{GpuStatus, MonitorSnapshot};
use ratatui::prelude::*;
use ratatui::widgets::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn render(
    frame: &mut Frame,
    app: &App,
    telemetry: Option<&IndexTelemetrySnapshot>,
    monitor: Option<&MonitorSnapshot>,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Footer/help
        ])
        .split(frame.area());

    render_header(frame, chunks[0], app);
    render_main(frame, chunks[1], app, telemetry, monitor);
    render_footer(frame, chunks[2], app);
}

fn render_header(frame: &mut Frame, area: Rect, app: &App) {
    let title = format!(
        " rmcp-memex wizard v{} - Step {}/{}: {} ",
        VERSION,
        app.step.step_number(),
        WizardStep::total_steps(),
        app.step.title()
    );

    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .title(title)
        .title_style(Style::default().fg(Color::Cyan).bold());

    frame.render_widget(block, area);
}

fn render_footer(frame: &mut Frame, area: Rect, app: &App) {
    let help_text = if app.input_mode || app.data_setup.input_mode {
        " [Enter] Confirm | [Esc] Cancel | Type to edit "
    } else {
        match app.step {
            WizardStep::Welcome => " [->/n] Next | [q] Quit ",
            WizardStep::EmbedderSetup => {
                " [Up/Down] Navigate | [Enter] Select | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::MemexSettings => {
                " [Up/Down] Navigate | [Enter] Edit | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::HostSelection => {
                " [Up/Down] Navigate | [Space] Toggle | [->] Next | [<-] Back | [q] Quit "
            }
            WizardStep::SnippetPreview => " [->] Next | [<-] Back | [q] Quit ",
            WizardStep::HealthCheck => {
                if app.health_running {
                    " Running health checks... "
                } else {
                    " [Enter/r] Run Check | [->] Next | [<-] Back | [q] Quit "
                }
            }
            WizardStep::DataSetup => match app.data_setup.sub_step {
                DataSetupSubStep::SelectOption => {
                    " [Up/Down] Select | [Enter] Choose | [->] Skip | [<-] Back | [q] Quit "
                }
                DataSetupSubStep::EnterPath | DataSetupSubStep::EnterNamespace => {
                    " [Enter] Confirm | [Esc] Cancel "
                }
                DataSetupSubStep::SelectImportMode => {
                    " [Up/Down] Select | [Enter] Choose | [<-] Back "
                }
                DataSetupSubStep::Indexing => {
                    let telemetry = app.telemetry_rx.as_ref().map(|rx| rx.borrow().clone());
                    if telemetry.as_ref().map(|s| s.complete).unwrap_or(false) {
                        " [->] Next | [<-] Back | [q] Quit "
                    } else if telemetry
                        .as_ref()
                        .map(|s| s.paused)
                        .unwrap_or(app.index_paused)
                    {
                        " [Space] Resume | [+] Par+ | [-] Par- | [s] Stop | [q] Quit "
                    } else {
                        " [Space] Pause | [+] Par+ | [-] Par- | [s] Stop | [q] Quit "
                    }
                }
                DataSetupSubStep::Complete => " [->] Next | [<-] Back | [q] Quit ",
            },
            WizardStep::Summary => " [Enter] Write Configs | [<-] Back | [q] Quit ",
        }
    };

    let help = Paragraph::new(help_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded),
        );

    frame.render_widget(help, area);
}

fn render_main(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    telemetry: Option<&IndexTelemetrySnapshot>,
    monitor: Option<&MonitorSnapshot>,
) {
    match app.step {
        WizardStep::Welcome => render_welcome(frame, area, app),
        WizardStep::EmbedderSetup => render_embedder_setup(frame, area, app),
        WizardStep::MemexSettings => render_settings(frame, area, app),
        WizardStep::HostSelection => render_host_selection(frame, area, app),
        WizardStep::SnippetPreview => render_snippet_preview(frame, area, app),
        WizardStep::HealthCheck => render_health_check(frame, area, app),
        WizardStep::DataSetup => render_data_setup(frame, area, app, telemetry, monitor),
        WizardStep::Summary => render_summary(frame, area, app),
    }
}

fn render_welcome(frame: &mut Frame, area: Rect, app: &App) {
    let text = vec![
        Line::from(""),
        Line::from(Span::styled(
            "Welcome to rmcp-memex Configuration Wizard",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from("This wizard will help you:"),
        Line::from(""),
        Line::from("  1. Configure rmcp-memex settings (database path, cache, etc.)"),
        Line::from("  2. Detect and configure MCP host integrations"),
        Line::from("  3. Generate configuration snippets for your hosts"),
        Line::from("  4. Verify your setup with a health check"),
        Line::from(""),
        Line::from(Span::styled(
            format!("Binary: {}", app.binary_path),
            Style::default().fg(Color::Green),
        )),
        Line::from(""),
        Line::from(format!(
            "Detected {} MCP host configuration(s)",
            app.hosts.iter().filter(|(_kind, h)| h.exists).count()
        )),
        Line::from(""),
        if app.dry_run {
            Line::from(Span::styled(
                "DRY RUN MODE - No files will be written",
                Style::default().fg(Color::Yellow).bold(),
            ))
        } else {
            Line::from("")
        },
        Line::from(""),
        Line::from(Span::styled(
            "Press → or 'n' to continue...",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let paragraph = Paragraph::new(text).alignment(Alignment::Center).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded),
    );

    frame.render_widget(paragraph, area);
}

fn settings_fields(app: &App) -> Vec<(&'static str, String, &'static str)> {
    let fields = vec![
        (
            "Database Path",
            app.memex_cfg.db_path.clone(),
            "LanceDB storage location",
        ),
        (
            "Path Mode",
            app.get_field_value(1),
            "shared or hostname-suffixed",
        ),
        (
            "HTTP/SSE Port",
            app.get_field_value(2),
            "disabled or a port number",
        ),
        (
            "Cache Size (MB)",
            app.get_field_value(3),
            "in-memory cache budget",
        ),
        (
            "Log Level",
            app.get_field_value(4),
            "trace/debug/info/warn/error",
        ),
        (
            "Max Request (bytes)",
            app.get_field_value(5),
            "JSON-RPC size limit",
        ),
        (
            "Deployment Mode",
            app.get_field_value(6),
            if app.mux_proxy_on_path() {
                "direct stdio or shared mux"
            } else {
                "shared mux requires rmcp_mux_proxy on PATH"
            },
        ),
    ];

    fields
}

fn render_settings(frame: &mut Frame, area: Rect, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(8)])
        .split(area);

    let info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Config path: ", Style::default().bold()),
            Span::raw(app.resolved_config_path()),
        ]),
        Line::from(vec![
            Span::styled("Effective database: ", Style::default().bold()),
            Span::raw(app.memex_cfg.resolved_db_path()),
        ]),
    ])
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Current Paths "),
    );
    frame.render_widget(info, chunks[0]);

    let fields = settings_fields(app);
    let items: Vec<ListItem> = fields
        .iter()
        .enumerate()
        .map(|(i, (label, value, hint))| {
            let is_focused = i == app.focus;
            let is_editing = app.input_mode && app.editing_field == Some(i);

            let display_value = if is_editing {
                format!("{}▏", app.input_buffer)
            } else {
                (*value).clone()
            };

            let style = if is_editing {
                Style::default().fg(Color::Yellow).bg(Color::DarkGray)
            } else if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else {
                Style::default()
            };

            let prefix = if is_focused { "▶ " } else { "  " };

            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(format!("{:<20}", label), style),
                Span::styled(format!("{:<40}", display_value), style),
                Span::styled(format!(" ({})", hint), Style::default().fg(Color::DarkGray)),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Memex Configuration "),
    );

    frame.render_widget(list, chunks[1]);
}

fn render_host_selection(frame: &mut Frame, area: Rect, app: &App) {
    let items: Vec<ListItem> = app
        .hosts
        .iter()
        .enumerate()
        .map(|(i, (kind, detection))| {
            let is_focused = i == app.focus;
            let is_selected = app.selected_hosts.contains(&i);

            let checkbox = if is_selected { "[x]" } else { "[ ]" };
            let status = detection.status_icon();

            let style = if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else if is_selected {
                Style::default().fg(Color::Green)
            } else if !detection.exists {
                Style::default().fg(Color::DarkGray)
            } else {
                Style::default()
            };

            let prefix = if is_focused { "▶ " } else { "  " };

            ListItem::new(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(format!("{} ", checkbox), style),
                Span::styled(format!("{} ", status), style),
                Span::styled(format!("{:<20}", kind.label()), style),
                Span::styled(
                    format!(" {} ", detection.status_text()),
                    if detection.has_rmcp_memex {
                        Style::default().fg(Color::Green)
                    } else if detection.exists {
                        Style::default().fg(Color::Yellow)
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                ),
            ]))
        })
        .collect();

    let list = List::new(items).block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .title(" Select MCP Hosts to Configure "),
    );

    frame.render_widget(list, area);
}

fn render_snippet_preview(frame: &mut Frame, area: Rect, app: &App) {
    let snippets = app.generate_snippets();

    let mut lines = vec![
        Line::from(Span::styled(
            "Configuration snippets for selected hosts:",
            Style::default().fg(Color::Cyan),
        )),
        Line::from(""),
    ];

    if snippets.is_empty() {
        lines.push(Line::from(Span::styled(
            "No hosts selected. Go back and select at least one host.",
            Style::default().fg(Color::Yellow),
        )));
    } else {
        for (kind, snippet) in snippets {
            lines.push(Line::from(Span::styled(
                format!("--- {} ---", kind.label()),
                Style::default().fg(Color::Green).bold(),
            )));
            for line in snippet.lines() {
                lines.push(Line::from(Span::styled(
                    format!("  {}", line),
                    Style::default().fg(Color::White),
                )));
            }
            lines.push(Line::from(""));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Config Snippets Preview "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_health_check(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Health Check",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
    ];

    // Show status line
    if app.health_running {
        lines.push(Line::from(Span::styled(
            "[...] Running health checks...",
            Style::default().fg(Color::Yellow),
        )));
    } else if let Some(status) = &app.health_status {
        lines.push(Line::from(Span::styled(
            status.clone(),
            if status.contains("passed") {
                Style::default().fg(Color::Green)
            } else if status.contains("failed") {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Yellow)
            },
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press [Enter] or [r] to run health checks",
            Style::default().fg(Color::Yellow),
        )));
    }

    lines.push(Line::from(""));

    // Show detailed health check results
    if let Some(ref result) = app.health_result {
        lines.push(Line::from(Span::styled("Checks:", Style::default().bold())));

        for item in &result.items {
            let (icon, color) = match &item.status {
                CheckStatus::Pass => ("[OK]", Color::Green),
                CheckStatus::Fail(_) => ("[ERR]", Color::Red),
                CheckStatus::Running => ("[...]", Color::Yellow),
                CheckStatus::Pending => ("[ ]", Color::DarkGray),
            };

            lines.push(Line::from(vec![
                Span::styled(format!("  {} ", icon), Style::default().fg(color)),
                Span::styled(&item.name, Style::default().fg(color)),
                Span::styled(
                    format!(" - {}", item.description),
                    Style::default().fg(Color::DarkGray),
                ),
            ]));

            // Show error details
            if let CheckStatus::Fail(ref msg) = item.status {
                for error_line in msg.lines() {
                    lines.push(Line::from(Span::styled(
                        format!("       {}", error_line),
                        Style::default().fg(Color::Red),
                    )));
                }
            }
        }

        lines.push(Line::from(""));

        // Show connected provider info
        if let Some(ref provider) = result.connected_provider {
            lines.push(Line::from(vec![
                Span::styled("  Connected to: ", Style::default()),
                Span::styled(provider, Style::default().fg(Color::Green).bold()),
            ]));
        }

        if let Some(dim) = result.verified_dimension {
            lines.push(Line::from(vec![
                Span::styled("  Vector dimension: ", Style::default()),
                Span::styled(dim.to_string(), Style::default().fg(Color::Cyan)),
            ]));
        }
    }

    // Show any additional messages
    if !app.messages.is_empty() {
        lines.push(Line::from(""));
        for msg in &app.messages {
            let style = if msg.starts_with("[OK]") {
                Style::default().fg(Color::Green)
            } else if msg.starts_with("[ERR]") {
                Style::default().fg(Color::Red)
            } else if msg.starts_with("[-]") {
                Style::default().fg(Color::Yellow)
            } else {
                Style::default()
            };
            lines.push(Line::from(Span::styled(msg.clone(), style)));
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Health Check "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_embedder_setup(frame: &mut Frame, area: Rect, app: &App) {
    let provider_label = if app.embedder_state.use_manual {
        "Manual configuration".to_string()
    } else if let Some(provider) = &app.embedder_state.selected_provider {
        provider.kind.label().to_string()
    } else {
        "No provider selected yet".to_string()
    };

    let model_label = app
        .embedder_state
        .selected_model()
        .unwrap_or_else(|| "<unset>".to_string());
    let base_url = app.embedder_state.selected_base_url().unwrap_or("<unset>");

    let mut lines = vec![
        Line::from(Span::styled(
            "Embedding Provider Setup",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from("Configure which embedding provider to use for vector search."),
        Line::from(""),
    ];

    // Show current configuration
    lines.push(Line::from(Span::styled(
        "Current Configuration:",
        Style::default().bold(),
    )));
    lines.push(Line::from(format!("  Provider: {}", provider_label)));
    lines.push(Line::from(format!("  Base URL: {}", base_url)));
    lines.push(Line::from(format!("  Model: {}", model_label)));
    lines.push(Line::from(format!(
        "  Dimension: {}",
        app.embedder_state.dimension_display()
    )));
    lines.push(Line::from(Span::styled(
        format!("             {}", app.embedder_state.dimension_hint()),
        Style::default().fg(Color::DarkGray),
    )));
    lines.push(Line::from(""));

    if app.embedder_state.use_manual {
        lines.push(Line::from(Span::styled(
            "Manual Fields:",
            Style::default().bold(),
        )));

        let fields = [
            ("Base URL", app.embedder_state.manual_url.clone()),
            ("Model", app.embedder_state.manual_model.clone()),
            ("Dimension", app.embedder_state.dimension_display()),
        ];

        for (i, (label, value)) in fields.iter().enumerate() {
            let is_focused = i == app.focus;
            let is_editing = app.input_mode && app.editing_field == Some(i);
            let display_value = if is_editing {
                format!("{}▏", app.input_buffer)
            } else {
                value.clone()
            };
            let style = if is_editing {
                Style::default().fg(Color::Yellow).bg(Color::DarkGray)
            } else if is_focused {
                Style::default().fg(Color::Cyan).bold()
            } else {
                Style::default()
            };
            let prefix = if is_focused { "▶ " } else { "  " };

            lines.push(Line::from(vec![
                Span::styled(prefix, style),
                Span::styled(format!("{:<12}", label), style),
                Span::styled(display_value, style),
            ]));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Press [Esc] to return to detected providers.",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        // Show detected providers if available
        if !app.embedder_state.detected_providers.is_empty() {
            lines.push(Line::from(Span::styled(
                "Detected Providers:",
                Style::default().bold(),
            )));

            for (i, provider) in app.embedder_state.detected_providers.iter().enumerate() {
                let is_focused = i == app.focus;
                let is_selected = app
                    .embedder_state
                    .selected_provider
                    .as_ref()
                    .map(|selected| {
                        selected.base_url == provider.base_url
                            && selected.model() == provider.model()
                    })
                    .unwrap_or(false);
                let prefix = if is_focused { "▶ " } else { "  " };
                let style = if is_focused {
                    Style::default().fg(Color::Cyan).bold()
                } else if is_selected {
                    Style::default().fg(Color::Green)
                } else {
                    Style::default()
                };

                let status_icon = match &provider.status {
                    ProviderStatus::Online(_) => "[OK]",
                    ProviderStatus::OnlineNoModel => "[--]",
                    ProviderStatus::Offline => "[XX]",
                };

                let status_color = match &provider.status {
                    ProviderStatus::Online(_) => Color::Green,
                    ProviderStatus::OnlineNoModel => Color::Yellow,
                    ProviderStatus::Offline => Color::DarkGray,
                };

                let dimension_note = "probe required".to_string();

                lines.push(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(
                        format!("{} ", status_icon),
                        Style::default().fg(status_color),
                    ),
                    Span::styled(provider.summary(), style),
                    Span::styled(
                        format!(" ({dimension_note})"),
                        Style::default().fg(Color::DarkGray),
                    ),
                ]));
            }
        } else {
            lines.push(Line::from(Span::styled(
                "No providers detected yet. Continue to proceed with default configuration.",
                Style::default().fg(Color::Yellow),
            )));
        }

        let manual_index = app.embedder_state.detected_providers.len();
        let manual_style = if app.focus == manual_index {
            Style::default().fg(Color::Cyan).bold()
        } else {
            Style::default()
        };
        let manual_prefix = if app.focus == manual_index {
            "▶ "
        } else {
            "  "
        };
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(manual_prefix, manual_style),
            Span::styled("Configure manually", manual_style),
            Span::styled(
                " (edit URL, model, and dimension directly)",
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Embedder Setup "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_data_setup(
    frame: &mut Frame,
    area: Rect,
    app: &App,
    telemetry: Option<&IndexTelemetrySnapshot>,
    monitor: Option<&MonitorSnapshot>,
) {
    if app.data_setup.sub_step == DataSetupSubStep::Indexing {
        render_indexing_dashboard(frame, area, telemetry, monitor);
        return;
    }

    let mut lines = vec![
        Line::from(Span::styled(
            "Data Setup",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
    ];

    match app.data_setup.sub_step {
        DataSetupSubStep::SelectOption => {
            lines.push(Line::from("Select how to initialize your data:"));
            lines.push(Line::from(""));

            for (i, option) in DataSetupOption::all().iter().enumerate() {
                let is_focused = i == app.data_setup.focus;
                let prefix = if is_focused { "▶ " } else { "  " };
                let style = if is_focused {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default()
                };

                lines.push(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(option.label(), style),
                ]));
                lines.push(Line::from(Span::styled(
                    format!("    {}", option.detail()),
                    Style::default().fg(Color::DarkGray),
                )));
            }
        }
        DataSetupSubStep::EnterPath => {
            lines.push(Line::from(format!(
                "Enter path for {:?}:",
                app.data_setup.option
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("> {}|", app.data_setup.input_buffer),
                Style::default().fg(Color::Yellow),
            )));
            if let Some(error) = &app.data_setup.validation_error {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    error.clone(),
                    Style::default().fg(Color::Red),
                )));
            }
        }
        DataSetupSubStep::EnterNamespace => {
            lines.push(Line::from("Enter namespace for indexed documents:"));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                format!("> {}|", app.data_setup.input_buffer),
                Style::default().fg(Color::Yellow),
            )));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "(Leave empty for default 'rag' namespace)",
                Style::default().fg(Color::DarkGray),
            )));
            if let Some(error) = &app.data_setup.validation_error {
                lines.push(Line::from(""));
                lines.push(Line::from(Span::styled(
                    error.clone(),
                    Style::default().fg(Color::Red),
                )));
            }
        }
        DataSetupSubStep::SelectImportMode => {
            lines.push(Line::from("Select import mode:"));
            lines.push(Line::from(""));

            for (i, mode) in ImportMode::all().iter().enumerate() {
                let is_focused = i == app.data_setup.focus;
                let prefix = if is_focused { "▶ " } else { "  " };
                let style = if is_focused {
                    Style::default().fg(Color::Cyan).bold()
                } else {
                    Style::default()
                };

                lines.push(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::styled(mode.label(), style),
                ]));
            }
        }
        DataSetupSubStep::Indexing => {}
        DataSetupSubStep::Complete => {
            lines.push(Line::from(Span::styled(
                "Data setup complete!",
                Style::default().fg(Color::Green).bold(),
            )));
            lines.push(Line::from(""));

            for msg in &app.messages {
                let style = if msg.starts_with("[OK]") {
                    Style::default().fg(Color::Green)
                } else if msg.starts_with("[ERR]") {
                    Style::default().fg(Color::Red)
                } else if msg.starts_with("[WARN]") {
                    Style::default().fg(Color::Yellow)
                } else {
                    Style::default()
                };
                lines.push(Line::from(Span::styled(msg.clone(), style)));
            }
        }
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Data Setup "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

fn render_indexing_dashboard(
    frame: &mut Frame,
    area: Rect,
    telemetry: Option<&IndexTelemetrySnapshot>,
    monitor: Option<&MonitorSnapshot>,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(10), Constraint::Length(4)])
        .split(area);
    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(outer[0]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(4), Constraint::Min(6)])
        .split(main[0]);

    let ratio = telemetry
        .map(|snapshot| {
            if snapshot.total == 0 {
                0.0
            } else {
                snapshot.processed as f64 / snapshot.total as f64
            }
        })
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);

    let progress_label = telemetry
        .map(|snapshot| format!("{}/{}", snapshot.processed, snapshot.total))
        .unwrap_or_else(|| "waiting".to_string());
    let progress_gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Index Progress "),
        )
        .gauge_style(Style::default().fg(Color::Cyan).bg(Color::DarkGray))
        .ratio(ratio)
        .label(progress_label);
    frame.render_widget(progress_gauge, left[0]);

    let left_lines = if let Some(snapshot) = telemetry {
        let status = if snapshot.complete {
            "complete"
        } else if snapshot.stopping {
            "stopping"
        } else if snapshot.paused {
            "paused"
        } else {
            "running"
        };
        vec![
            Line::from(format!(
                "Rate: {:.2} files/sec | Avg Embed: {}",
                snapshot.files_per_sec,
                snapshot.avg_embedder_ms.map(|ms| format!("{ms:.0}ms")).unwrap_or_else(|| "--".to_string())
            )),
            Line::from(format!(
                "ETA: {} | Tokens: {}",
                format_eta(snapshot.eta_secs),
                snapshot.total_tokens_estimated
            )),
            Line::from(format!("State: {}", status)),
            Line::from(format!(
                "Parallelism: {} | Inflight: {}",
                snapshot.parallelism, snapshot.in_flight
            )),
            Line::from(format!(
                "Indexed: {} | Skipped: {} | Failed: {}",
                snapshot.indexed, snapshot.skipped, snapshot.failed
            )),
            Line::from(format!("Current file: {}", current_file_label(snapshot))),
        ]
    } else {
        vec![
            Line::from("Waiting for indexing telemetry..."),
            Line::from(""),
            Line::from("The scheduler will publish live stats here."),
        ]
    };
    let left_stats = Paragraph::new(left_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Operator Stats "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(left_stats, left[1]);

    let right_lines = if let Some(snapshot) = monitor {
        let gpu_status = match &snapshot.gpu_status {
            GpuStatus::Available { class_name } => format!("available ({class_name})"),
            GpuStatus::Unavailable { reason } => format!("unavailable ({reason})"),
        };
        vec![
            Line::from(format!("System CPU: {:.1}%", snapshot.system_cpu_percent)),
            Line::from(format!(
                "System RAM: {} / {}",
                MonitorSnapshot::format_bytes(snapshot.system_ram_used),
                MonitorSnapshot::format_bytes(snapshot.system_ram_total)
            )),
            Line::from(format!("rmcp-memex CPU: {:.1}%", snapshot.rmcp_memex_cpu)),
            Line::from(format!(
                "rmcp-memex RSS: {}",
                MonitorSnapshot::format_bytes(snapshot.rmcp_memex_rss)
            )),
            Line::from(format!(
                "Embedder CPU: {:.1}%",
                snapshot.embedder_cpu_aggregate
            )),
            Line::from(format!(
                "Embedder RSS: {}",
                MonitorSnapshot::format_bytes(snapshot.embedder_rss_aggregate)
            )),
            Line::from(format!(
                "GPU util: {}",
                snapshot
                    .gpu_util_percent
                    .map(|value| format!("{value:.1}%"))
                    .unwrap_or_else(|| "--".to_string())
            )),
            Line::from(format!(
                "GPU memory: {} / {}",
                snapshot
                    .gpu_memory_used
                    .map(MonitorSnapshot::format_bytes)
                    .unwrap_or_else(|| "--".to_string()),
                snapshot
                    .gpu_memory_total
                    .map(MonitorSnapshot::format_bytes)
                    .unwrap_or_else(|| "--".to_string())
            )),
            Line::from(format!("GPU status: {gpu_status}")),
        ]
    } else {
        vec![
            Line::from("Waiting for system telemetry..."),
            Line::from(""),
            Line::from("CPU, RAM, process, and GPU stats will stream here."),
        ]
    };
    let right_stats = Paragraph::new(right_lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" System Telemetry "),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(right_stats, main[1]);

    let warning_lines = if let Some(snapshot) = telemetry {
        let lines: Vec<Line> = snapshot
            .recent_warnings
            .iter()
            .rev()
            .take(3)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|warning| {
                Line::from(Span::styled(
                    format!("[{}] {}", warning.code, warning.message),
                    Style::default().fg(Color::Yellow),
                ))
            })
            .collect();
        if lines.is_empty() {
            vec![Line::from(Span::styled(
                "No warnings.",
                Style::default().fg(Color::DarkGray),
            ))]
        } else {
            lines
        }
    } else {
        vec![Line::from(Span::styled(
            "Waiting for warnings...",
            Style::default().fg(Color::DarkGray),
        ))]
    };
    let warnings = Paragraph::new(warning_lines).wrap(Wrap { trim: false });
    frame.render_widget(warnings, outer[1]);
}

fn current_file_label(snapshot: &IndexTelemetrySnapshot) -> String {
    snapshot
        .current_file
        .clone()
        .unwrap_or_else(|| "none".to_string())
}

fn format_eta(eta_secs: Option<f64>) -> String {
    let Some(eta_secs) = eta_secs else {
        return "--".to_string();
    };

    let eta_secs = eta_secs.max(0.0).round() as u64;
    let minutes = eta_secs / 60;
    let seconds = eta_secs % 60;
    if minutes > 0 {
        format!("{minutes}m {seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn render_summary(frame: &mut Frame, area: Rect, app: &App) {
    let mut lines = vec![
        Line::from(Span::styled(
            "Configuration Summary",
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled("Memex Settings:", Style::default().bold())),
        Line::from(format!("  Config: {}", app.config_path)),
        Line::from(format!("  Database: {}", app.memex_cfg.resolved_db_path())),
        Line::from(format!("  Cache: {} MB", app.memex_cfg.cache_mb)),
        Line::from(format!("  Log Level: {}", app.memex_cfg.log_level)),
        Line::from(""),
        Line::from(Span::styled("Selected Hosts:", Style::default().bold())),
    ];

    let selected = app.get_selected_hosts();
    if selected.is_empty() {
        lines.push(Line::from(Span::styled(
            "  (none selected)",
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        for (kind, _detection) in selected {
            lines.push(Line::from(format!("  • {}", kind.label())));
        }
    }

    lines.push(Line::from(""));

    if app.dry_run {
        lines.push(Line::from(Span::styled(
            "DRY RUN MODE - Press [Enter] to preview changes",
            Style::default().fg(Color::Yellow).bold(),
        )));
    } else {
        lines.push(Line::from(Span::styled(
            "Press [Enter] to write configuration files",
            Style::default().fg(Color::Green).bold(),
        )));
    }

    lines.push(Line::from(""));

    // Show any messages from write operation
    for msg in &app.messages {
        let style = if msg.starts_with('✓') {
            Style::default().fg(Color::Green)
        } else if msg.contains("Warning") || msg.contains("DRY RUN") {
            Style::default().fg(Color::Yellow)
        } else if msg.contains("Error") {
            Style::default().fg(Color::Red)
        } else {
            Style::default().fg(Color::White)
        };
        lines.push(Line::from(Span::styled(msg.clone(), style)));
    }

    let paragraph = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .title(" Summary "),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    use super::settings_fields;
    use crate::tui::app::{App, WizardConfig};

    #[test]
    fn settings_fields_track_live_app_field_count() {
        let app = App::new(WizardConfig::default());
        assert_eq!(settings_fields(&app).len(), app.settings_field_count());
    }

    #[test]
    fn settings_fields_keep_deployment_mode_visible_without_mux_proxy() {
        let mut app = App::new(WizardConfig::default());
        app.mux_proxy_command = None;

        let fields = settings_fields(&app);
        assert_eq!(fields.len(), 7);
        assert_eq!(fields[6].0, "Deployment Mode");
        assert_eq!(fields[6].1, "Per-host (shared unavailable)".to_string());
    }
}
