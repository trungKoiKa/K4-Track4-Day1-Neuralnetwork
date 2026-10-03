"""Run the same prospective stages as the submitted Colab notebook."""
from protocol import stage
from revision_report import finish
if __name__ == '__main__':
    for name in ['setup', 'health', 'pilots', 'baseline', 'experiments', 'select', 'score']:
        stage(name)
    finish()
